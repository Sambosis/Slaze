#!/usr/bin/env python3
"""
ast_code_editor.py — AST-perfect Python code editor for LLM agents (no embeddings).

Key features:
  • AST-accurate chunking of Python: module / function / class / method
  • Robust, line-anchored edits: replace whole def, body-only, docstring-only, insert-after, delete
  • Deterministic guardrails: re-parse validation before write; atomic file replace
  • StringEditor for in-memory use; FileEditor for on-disk edits
  • Simple CLI with dry-run diffs

Examples (library):
  from ast_code_editor import FileEditor

  ed = FileEditor("/path/to/repo")
  # Preview available symbols in a file
  chunks = ed.list_symbols("pkg/example.py")
  # Replace a function body (keep existing docstring)
  ed.replace_body("pkg/example.py", "top_function", "return 3 * x")
  # Insert a new function after an existing one (module-level)
  ed.insert_after("pkg/example.py", "top_function", "def new_util(y):\n    return y*y\n")
  # Delete a method
  ed.delete_symbol("pkg/example.py", "Greeter.greet")

Examples (CLI):
  python ast_code_editor.py list pkg/example.py
  python ast_code_editor.py show pkg/example.py Greeter.greet
  python ast_code_editor.py replace-body pkg/example.py top_function --text "return 3 * x" --dry-run
  python ast_code_editor.py replace-whole pkg/example.py Greeter --file new_class_impl.py
  python ast_code_editor.py replace-docstring pkg/example.py Greeter.greet --text "Return a greeting."
  python ast_code_editor.py insert-after pkg/example.py top_function --file add_me.py
  python ast_code_editor.py delete pkg/example.py Greeter.greet --dry-run
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import tokenize

# ----------------------------
# AST Chunker (Python only)
# ----------------------------

Span = Tuple[int, int]  # inclusive 1-based line span: (start_line, end_line)

@dataclass
class Chunk:
    kind: str           # "module" | "function" | "class" | "method"
    symbol: str         # e.g., "module" | "foo" | "MyClass" | "MyClass.bar"
    span: Span          # inclusive line span
    text: str
    file_path: str
    chunk_index: int

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", ".tox", ".mypy_cache", "__pycache__", ".pytest_cache",
    "node_modules", "dist", "build", "out", ".next", ".nuxt", ".vercel", ".cache",
    ".idea", ".vscode", ".venv", "venv", "env",
    "target", "bin", "obj", ".gradle", ".terraform", ".mvn",
}

def _pep263_read(path: Path) -> Optional[str]:
    try:
        with tokenize.open(path) as f:
            return f.read()
    except Exception:
        return None

def _safe_get_end_lineno(src: str, node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if isinstance(end, int):
        return end
    seg = ast.get_source_segment(src, node)
    if seg is not None:
        return node.lineno + seg.count("\n")
    last = None
    for child in ast.walk(node):
        if hasattr(child, "lineno"):
            cend = getattr(child, "end_lineno", getattr(child, "lineno", None))
            if isinstance(cend, int):
                if last is None or cend > last:
                    last = cend
    return last if isinstance(last, int) else node.lineno

def _span_for(node: ast.AST, src: str, include_decorators: bool = True) -> Span:
    start = node.lineno
    if include_decorators and hasattr(node, "decorator_list"):
        decos = getattr(node, "decorator_list", [])
        if decos:
            start = min(start, min(d.lineno for d in decos))
    end = _safe_get_end_lineno(src, node)
    return (start, end)

def _slice_lines(lines: List[str], span: Span) -> str:
    s, e = span
    s = max(1, s); e = max(s, e)
    return "".join(lines[s-1:e])

def _nonempty(text: str) -> bool:
    return any(line.strip() for line in text.splitlines())

def _subtract(parent: Span, children: List[Span]) -> List[Span]:
    s, e = parent
    if not children:
        return [(s, e)]
    ch = sorted([(max(s, cs), min(e, ce)) for cs, ce in children if not (ce < s or cs > e)])
    out: List[Span] = []
    cursor = s
    for cs, ce in ch:
        if cs > cursor:
            out.append((cursor, cs - 1))
        cursor = max(cursor, ce + 1)
    if cursor <= e:
        out.append((cursor, e))
    return out

def iter_python_chunks_for_text(src: str, file_path: str = "<memory>") -> Iterator[Chunk]:
    lines = src.splitlines(keepends=True)
    try:
        tree = ast.parse(src, filename=file_path)
    except SyntaxError:
        # Fallback: whole file as a single module chunk
        text = "".join(lines)
        if _nonempty(text):
            yield Chunk("module", "module", (1, len(lines)), text, file_path, 0)
        return

    idx = 0
    top_spans: List[Span] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sp = _span_for(node, src, include_decorators=True)
            text = _slice_lines(lines, sp)
            if _nonempty(text):
                yield Chunk("function", node.name, sp, text, file_path, idx); idx += 1
            top_spans.append(sp)

        elif isinstance(node, ast.ClassDef):
            class_span = _span_for(node, src, include_decorators=True)
            top_spans.append(class_span)

            inner_spans: List[Span] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sp = _span_for(child, src, include_decorators=True)
                    text = _slice_lines(lines, sp)
                    if _nonempty(text):
                        yield Chunk("method", f"{node.name}.{child.name}", sp, text, file_path, idx); idx += 1
                    inner_spans.append(sp)
                elif isinstance(child, ast.ClassDef):
                    sp = _span_for(child, src, include_decorators=True)
                    text = _slice_lines(lines, sp)
                    if _nonempty(text):
                        yield Chunk("class", f"{node.name}.{child.name}", sp, text, file_path, idx); idx += 1
                    inner_spans.append(sp)

            # class-only leftovers (header/docstring/attrs), without methods/nested classes
            for sp in _subtract(class_span, inner_spans):
                text = _slice_lines(lines, sp)
                if _nonempty(text):
                    yield Chunk("class", node.name, sp, text, file_path, idx); idx += 1

    # Module leftovers (imports, globals, main-guard, etc.)
    for sp in _subtract((1, len(lines)), top_spans):
        text = _slice_lines(lines, sp)
        if _nonempty(text):
            yield Chunk("module", "module", sp, text, file_path, idx); idx += 1

def iter_python_chunks_for_file(path: Path) -> Iterator[Chunk]:
    src = _pep263_read(path)
    if src is None:
        return
    yield from iter_python_chunks_for_text(src, str(path))

# ----------------------------
# Editing primitives
# ----------------------------

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

@dataclass
class Replacement:
    start_index: int
    end_index: int
    text: str

def _line_to_abs_offsets(lines: List[str], span: Span) -> Tuple[int, int]:
    """Convert 1-based inclusive line span to [start_idx, end_idx) absolute offsets in the joined text."""
    s, e = span
    s = max(1, s); e = max(s, e)
    prefix = sum(len(l) for l in lines[:s-1])
    end = sum(len(l) for l in lines[:e])  # end is exclusive
    return prefix, end

def _apply_replacements(base_text: str, repls: List[Replacement]) -> str:
    """Apply non-overlapping replacements sorted by descending start_index."""
    buf = base_text
    for r in sorted(repls, key=lambda r: (r.start_index, r.end_index), reverse=True):
        buf = buf[:r.start_index] + r.text + buf[r.end_index:]
    return buf

def _validate_python(text: str, filename: str = "<edited>") -> None:
    ast.parse(text, filename=filename)

def _indent_like(sample_line: str) -> str:
    return sample_line[:len(sample_line) - len(sample_line.lstrip(" \t"))]

def _indent_block(block: str, indent: str) -> str:
    if not block.endswith("\n"):
        block += "\n"
    return "".join((indent + line if line.strip() else line) for line in block.splitlines(keepends=True))

# ----------------------------
# Editors
# ----------------------------

class StringEditor:
    """Edit a Python source string via AST-anchored operations."""
    def __init__(self, text: str, filename: str = "<memory>"):
        self.filename = filename
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    def list_symbols(self) -> List[Chunk]:
        return list(iter_python_chunks_for_text(self._text, self.filename))

    def _find_chunk(self, symbol: str) -> Chunk:
        matches = [c for c in self.list_symbols() if c.symbol == symbol]
        if not matches:
            raise ValueError(f"Symbol not found: {symbol}")
        # Prefer most specific: exact match with kind precedence method>function>class>module
        kind_rank = {"method": 3, "function": 2, "class": 1, "module": 0}
        matches.sort(key=lambda c: (kind_rank.get(c.kind, -1), c.span[0], -c.span[1]), reverse=True)
        return matches[0]

    def replace_whole(self, symbol: str, new_text: str, validate: bool = True) -> None:
        """Replace entire def/class including decorators and signature."""
        lines = self._text.splitlines(keepends=True)
        chunk = self._find_chunk(symbol)
        start, end = _line_to_abs_offsets(lines, chunk.span)
        new_src = self._text[:start] + _ensure_trailing_nl(new_text) + self._text[end:]
        if validate:
            _validate_python(new_src, self.filename)
        self._text = new_src

    def replace_docstring(self, symbol: str, new_docstring: str, triple: str = '"""', validate: bool = True) -> None:
        """Replace only the docstring of a function/class/method. Creates one if missing."""
        src = self._text
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=self.filename)

        # Navigate to node
        target = _resolve_symbol_node(tree, symbol)
        if target is None:
            raise ValueError(f"Symbol not found: {symbol}")

        # Determine docstring node span (if present)
        doc_span = _docstring_span(src, target)
        indent = _indent_for_body(lines, target)

        doc_block = f'{indent}{triple}{new_docstring}{triple}\n'
        repls: List[Replacement] = []

        if doc_span:
            s_idx, e_idx = _line_to_abs_offsets(lines, doc_span)
            repls.append(Replacement(s_idx, e_idx, doc_block))
        else:
            # Insert as first statement after header
            insertion_line = _first_body_linenum(target) or (getattr(target, "lineno", 1) + 1)
            ins_idx = sum(len(l) for l in lines[:insertion_line-1])
            repls.append(Replacement(ins_idx, ins_idx, doc_block))

        new_src = _apply_replacements(src, repls)
        if validate:
            _validate_python(new_src, self.filename)
        self._text = new_src

    def replace_body(self, symbol: str, new_body: str, keep_docstring: bool = True, validate: bool = True) -> None:
        """Replace just the body of a function/class/method (preserves header/defs)."""
        src = self._text
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=self.filename)

        target = _resolve_symbol_node(tree, symbol)
        if target is None:
            raise ValueError(f"Symbol not found: {symbol}")

        # Compute current body span (including docstring if keep_docstring=True)
        body_span = _body_span(src, target, include_docstring=keep_docstring)
        if body_span is None:
            # Empty body; insert after header indentation
            body_start_line = (getattr(target, "lineno", 1) + 1)
            body_span = (body_start_line, body_start_line - 1)  # empty range

        indent = _indent_for_body(lines, target)
        new_block = _indent_block(new_body, indent)
        s_idx, e_idx = _line_to_abs_offsets(lines, body_span)
        new_src = src[:s_idx] + new_block + src[e_idx:]
        if validate:
            _validate_python(new_src, self.filename)
        self._text = new_src

    def delete_symbol(self, symbol: str, validate: bool = True) -> None:
        """Delete a function/class/method (including decorators)."""
        src = self._text
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=self.filename)
        target = _resolve_symbol_node(tree, symbol)
        if target is None:
            raise ValueError(f"Symbol not found: {symbol}")

        span = _span_for(target, src, include_decorators=True)
        s_idx, e_idx = _line_to_abs_offsets(lines, span)
        # Also remove trailing blank lines directly following the span (nice hygiene)
        tail = src[e_idx:]
        rm_extra = 0
        for ch in tail.splitlines(keepends=True):
            if ch.strip():
                break
            rm_extra += len(ch)
        new_src = src[:s_idx] + src[e_idx + rm_extra:]
        if validate:
            _validate_python(new_src, self.filename)
        self._text = new_src

    def insert_after(self, symbol: str, new_text: str, validate: bool = True) -> None:
        """Insert a new def/class *after* the given symbol, in the same lexical block (module or class body)."""
        src = self._text
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=self.filename)
        target = _resolve_symbol_node(tree, symbol)
        if target is None:
            raise ValueError(f"Symbol not found: {symbol}")

        # Compute insertion after target span
        span = _span_for(target, src, include_decorators=True)
        _, e_idx = _line_to_abs_offsets(lines, span)
        # Add a separating newline if needed
        pad = "" if src[e_idx-1:e_idx] == "\n" else "\n"
        insert_text = pad + _ensure_trailing_nl(new_text)
        new_src = src[:e_idx] + insert_text + src[e_idx:]
        if validate:
            _validate_python(new_src, self.filename)
        self._text = new_src

    # Utilities
    def diff(self, other_text: str) -> str:
        return "".join(difflib.unified_diff(
            self._text.splitlines(keepends=True),
            other_text.splitlines(keepends=True),
            fromfile="before", tofile="after"
        ))

class FileEditor:
    """Edit files on disk atomically; re-parse validation enforced."""
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ----- Listing / viewing -----
    def list_symbols(self, rel_path: str | Path) -> List[Chunk]:
        p = self.root / rel_path
        src = _pep263_read(p)
        if src is None:
            return []
        return list(iter_python_chunks_for_text(src, str(p)))

    def show(self, rel_path: str | Path, symbol: str) -> str:
        p = self.root / rel_path
        src = _pep263_read(p) or ""
        ed = StringEditor(src, str(p))
        return ed._find_chunk(symbol).text

    # ----- Mutations -----
    def replace_whole(self, rel_path: str | Path, symbol: str, new_text: str, dry_run: bool = False) -> Optional[str]:
        return self._apply(rel_path, lambda ed: ed.replace_whole(symbol, new_text), dry_run)

    def replace_docstring(self, rel_path: str | Path, symbol: str, new_docstring: str, triple: str = '"""', dry_run: bool = False) -> Optional[str]:
        return self._apply(rel_path, lambda ed: ed.replace_docstring(symbol, new_docstring, triple), dry_run)

    def replace_body(self, rel_path: str | Path, symbol: str, new_body: str, keep_docstring: bool = True, dry_run: bool = False) -> Optional[str]:
        return self._apply(rel_path, lambda ed: ed.replace_body(symbol, new_body, keep_docstring), dry_run)

    def delete_symbol(self, rel_path: str | Path, symbol: str, dry_run: bool = False) -> Optional[str]:
        return self._apply(rel_path, lambda ed: ed.delete_symbol(symbol), dry_run)

    def insert_after(self, rel_path: str | Path, symbol: str, new_text: str, dry_run: bool = False) -> Optional[str]:
        return self._apply(rel_path, lambda ed: ed.insert_after(symbol, new_text), dry_run)

    # ----- Core apply -----
    def _apply(self, rel_path: str | Path, fn, dry_run: bool) -> Optional[str]:
        p = self.root / rel_path
        src = _pep263_read(p)
        if src is None:
            raise FileNotFoundError(f"Cannot read file: {p}")
        ed = StringEditor(src, str(p))
        before = ed.text
        fn(ed)  # may raise ValueError/SyntaxError with clear message
        after = ed.text

        if dry_run:
            return ed.diff(after)

        # Atomic write
        tmp = p.with_suffix(p.suffix + f".tmp.{_sha1(after)[:8]}")
        tmp.write_text(after, encoding="utf-8")
        os.replace(tmp, p)  # atomic on POSIX/Windows
        return None

# ----------------------------
# AST helpers for body/docstrings and symbol resolution
# ----------------------------

def _ensure_trailing_nl(s: str) -> str:
    return s if s.endswith("\n") else s + "\n"

def _resolve_symbol_node(tree: ast.Module, symbol: str) -> Optional[ast.AST]:
    """Resolve 'foo', 'Class', 'Class.method', 'Outer.Inner', 'Outer.Inner.method'."""
    parts = symbol.split(".")
    def walk(scope_body: List[ast.stmt], idx: int) -> Optional[ast.AST]:
        if idx >= len(parts):
            return None
        name = parts[idx]
        for stmt in scope_body:
            if isinstance(stmt, ast.ClassDef) and stmt.name == name:
                if idx == len(parts) - 1:
                    return stmt
                return walk(stmt.body, idx + 1)
            if idx == len(parts) - 1:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == name:
                    return stmt
        return None
    return walk(tree.body, 0)

def _first_body_linenum(node: ast.AST) -> Optional[int]:
    body = getattr(node, "body", None)
    if not body:
        return None
    return getattr(body[0], "lineno", None)

def _docstring_span(src: str, node: ast.AST) -> Optional[Span]:
    """Return span of the docstring triple-quoted literal if present as first body statement."""
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
        # Use exact span of the docstring expression
        return (_span_for(first, src, include_decorators=False))
    return None

def _indent_for_body(lines: List[str], node: ast.AST) -> str:
    """Indentation for the *body* of a def/class based on the first body line (fallback from header)."""
    fn_line = getattr(node, "lineno", 1)
    # Prefer first actual body line (after docstring if present)
    body_line = _first_body_linenum(node)
    if body_line is None:
        # Guess: indent from header line + 1
        body_line = fn_line + 1 if fn_line < len(lines) else len(lines)
    sample = lines[min(max(1, body_line) - 1, len(lines) - 1)]
    return sample[:len(sample) - len(sample.lstrip(" \t"))]

def _body_span(src: str, node: ast.AST, include_docstring: bool = True) -> Optional[Span]:
    """Span for the body region to be replaced (excludes signature/decorators)."""
    body = getattr(node, "body", None)
    if not body:
        return None
    # If docstring present and kept, include it in the range
    start_node = body[0] if include_docstring else (body[1] if len(body) > 1 and _is_docstring_node(body[0]) else body[0])
    start = getattr(start_node, "lineno", getattr(node, "lineno", 1) + 1)
    end = _safe_get_end_lineno(src, body[-1])
    return (start, end)

def _is_docstring_node(n: ast.AST) -> bool:
    return isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant) and isinstance(n.value.value, str)

# ----------------------------
# CLI
# ----------------------------

def _cmd_list(args):
    ed = FileEditor(args.root)
    chunks = ed.list_symbols(args.path)
    for c in chunks:
        print(f"{c.kind:7} | {c.symbol:30} | {c.span[0]:>5}-{c.span[1]:<5} | {Path(c.file_path).name}")

def _cmd_show(args):
    ed = FileEditor(args.root)
    print(ed.show(args.path, args.symbol), end="")

def _maybe_read_text_arg(args) -> str:
    if args.text is not None:
        return args.text if args.text.endswith("\n") else args.text + "\n"
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8")
    raise SystemExit("You must provide --text or --file")

def _print_or_apply(ed_method, dry_run: bool):
    diff = ed_method(dry_run=dry_run)
    if dry_run:
        if diff.strip():
            print(diff, end="")
        else:
            print("# (no changes)")

def _cmd_replace_whole(args):
    ed = FileEditor(args.root)
    new_text = _maybe_read_text_arg(args)
    _print_or_apply(lambda dry_run: ed.replace_whole(args.path, args.symbol, new_text, dry_run=dry_run), args.dry_run)

def _cmd_replace_body(args):
    ed = FileEditor(args.root)
    new_body = _maybe_read_text_arg(args)
    _print_or_apply(lambda dry_run: ed.replace_body(args.path, args.symbol, new_body, keep_docstring=not args.drop_docstring, dry_run=dry_run), args.dry_run)

def _cmd_replace_docstring(args):
    ed = FileEditor(args.root)
    text = _maybe_read_text_arg(args).rstrip("\n")
    _print_or_apply(lambda dry_run: ed.replace_docstring(args.path, args.symbol, text, triple=args.quote, dry_run=dry_run), args.dry_run)

def _cmd_delete(args):
    ed = FileEditor(args.root)
    _print_or_apply(lambda dry_run: ed.delete_symbol(args.path, args.symbol, dry_run=dry_run), args.dry_run)

def _cmd_insert_after(args):
    ed = FileEditor(args.root)
    new_text = _maybe_read_text_arg(args)
    _print_or_apply(lambda dry_run: ed.insert_after(args.path, args.symbol, new_text, dry_run=dry_run), args.dry_run)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AST-perfect Python code editor")
    p.add_argument("root", type=str, help="Project root (base dir)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List symbols in file")
    p_list.add_argument("path", type=str, help="File path relative to root")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Show a symbol source")
    p_show.add_argument("path", type=str)
    p_show.add_argument("symbol", type=str)
    p_show.set_defaults(func=_cmd_show)

    def add_text_file_opts(sp):
        sp.add_argument("--text", type=str, default=None, help="Inline text content")
        sp.add_argument("--file", type=str, default=None, help="Read content from file")
        sp.add_argument("--dry-run", action="store_true", help="Validate and show diff, but do not write")

    p_whole = sub.add_parser("replace-whole", help="Replace entire def/class (incl. decorators)")
    p_whole.add_argument("path", type=str)
    p_whole.add_argument("symbol", type=str)
    add_text_file_opts(p_whole)
    p_whole.set_defaults(func=_cmd_replace_whole)

    p_body = sub.add_parser("replace-body", help="Replace the body of a def/class/method")
    p_body.add_argument("path", type=str)
    p_body.add_argument("symbol", type=str)
    p_body.add_argument("--drop-docstring", action="store_true", help="Do not preserve existing docstring")
    add_text_file_opts(p_body)
    p_body.set_defaults(func=_cmd_replace_body)

    p_doc = sub.add_parser("replace-docstring", help="Replace or insert a docstring")
    p_doc.add_argument("path", type=str)
    p_doc.add_argument("symbol", type=str)
    p_doc.add_argument("--quote", type=str, default='"""', choices=['"""', "'''"], help="Docstring quotes")
    add_text_file_opts(p_doc)
    p_doc.set_defaults(func=_cmd_replace_docstring)

    p_del = sub.add_parser("delete", help="Delete a def/class/method")
    p_del.add_argument("path", type=str)
    p_del.add_argument("symbol", type=str)
    p_del.add_argument("--dry-run", action="store_true")
    p_del.set_defaults(func=_cmd_delete)

    p_ins = sub.add_parser("insert-after", help="Insert new def/class after a symbol")
    p_ins.add_argument("path", type=str)
    p_ins.add_argument("symbol", type=str)
    add_text_file_opts(p_ins)
    p_ins.set_defaults(func=_cmd_insert_after)

    return p.parse_args()

def main():
    args = _parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
