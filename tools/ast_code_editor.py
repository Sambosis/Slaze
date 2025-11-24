from __future__ import annotations

from enum import Enum
from typing import Literal, List, Optional, Tuple, Dict
from pathlib import Path
import os
import ast
import difflib
import hashlib
import tokenize
import logging

# type: ignore[override]
from .base import ToolResult, BaseAnthropicTool
from config import get_constant

from rich import print as rr  # align with existing tool
logger = logging.getLogger(__name__)

# ----------------------------
# Utility & Safety
# ----------------------------

def _ensure_trailing_nl(s: str) -> str:
    return s if s.endswith("\n") else s + "\n"

def _pep263_read(path: Path) -> Optional[str]:
    try:
        with tokenize.open(path) as f:
            return f.read()
    except Exception:
        return None

def _within_repo(repo: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(repo.resolve())
        return True
    except Exception:
        return False

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()




# ----------------------------
# AST Chunker (Python-only)
# ----------------------------

Span = Tuple[int, int]  # inclusive 1-based (start_line, end_line)

class Chunk(Dict):
    # "kind": "module" | "function" | "class" | "method"
    # "symbol": str                # e.g. module | foo | Class | Class.method
    # "span": (start_line, end_line)
    # "text": str
    # "file_path": str
    # "chunk_index": int
    pass

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

def _iter_chunks_for_text(src: str, file_path: str = "<memory>") -> List[Chunk]:
    lines = src.splitlines(keepends=True)
    try:
        tree = ast.parse(src, filename=file_path)
    except SyntaxError:
        text = "".join(lines)
        return [{
            "kind": "module", "symbol": "module",
            "span": (1, len(lines)), "text": text,
            "file_path": file_path, "chunk_index": 0
        }] if _nonempty(text) else []

    idx = 0
    out: List[Chunk] = []
    top_spans: List[Span] = []

    def _recurse(node: ast.AST, parent_symbol: str):
        nonlocal idx
        # This helper is for finding nested functions/classes inside other functions/classes
        # It does NOT subtract spans, it just adds them.
        if not hasattr(node, "body"): return
        
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sym = f"{parent_symbol}.{child.name}"
                sp = _span_for(child, src, include_decorators=True)
                text = _slice_lines(lines, sp)
                if _nonempty(text):
                    out.append({"kind":"function","symbol":sym,"span":sp,"text":text,"file_path":file_path,"chunk_index":idx})
                    idx += 1
                _recurse(child, sym)
            elif isinstance(child, ast.ClassDef):
                sym = f"{parent_symbol}.{child.name}"
                sp = _span_for(child, src, include_decorators=True)
                text = _slice_lines(lines, sp)
                if _nonempty(text):
                    out.append({"kind":"class","symbol":sym,"span":sp,"text":text,"file_path":file_path,"chunk_index":idx})
                    idx += 1
                _recurse(child, sym)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sp = _span_for(node, src, include_decorators=True)
            text = _slice_lines(lines, sp)
            if _nonempty(text):
                out.append({"kind":"function","symbol":node.name,"span":sp,"text":text,"file_path":file_path,"chunk_index":idx}); idx += 1
            top_spans.append(sp)
            _recurse(node, node.name)

        elif isinstance(node, ast.ClassDef):
            class_span = _span_for(node, src, include_decorators=True)
            top_spans.append(class_span)

            inner_spans: List[Span] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sp = _span_for(child, src, include_decorators=True)
                    text = _slice_lines(lines, sp)
                    if _nonempty(text):
                        out.append({"kind":"method","symbol":f"{node.name}.{child.name}","span":sp,"text":text,"file_path":file_path,"chunk_index":idx}); idx += 1
                    inner_spans.append(sp)
                    _recurse(child, f"{node.name}.{child.name}")
                elif isinstance(child, ast.ClassDef):
                    sp = _span_for(child, src, include_decorators=True)
                    text = _slice_lines(lines, sp)
                    if _nonempty(text):
                        out.append({"kind":"class","symbol":f"{node.name}.{child.name}","span":sp,"text":text,"file_path":file_path,"chunk_index":idx}); idx += 1
                    inner_spans.append(sp)
                    _recurse(child, f"{node.name}.{child.name}")

            for sp in _subtract(class_span, inner_spans):
                text = _slice_lines(lines, sp)
                if _nonempty(text):
                    out.append({"kind":"class","symbol":node.name,"span":sp,"text":text,"file_path":file_path,"chunk_index":idx}); idx += 1
        
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    sp = _span_for(node, src)
                    text = _slice_lines(lines, sp)
                    if _nonempty(text):
                        out.append({"kind":"variable","symbol":t.id,"span":sp,"text":text,"file_path":file_path,"chunk_index":idx}); idx += 1
                    top_spans.append(sp)
        
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                sp = _span_for(node, src)
                text = _slice_lines(lines, sp)
                if _nonempty(text):
                    out.append({"kind":"variable","symbol":node.target.id,"span":sp,"text":text,"file_path":file_path,"chunk_index":idx}); idx += 1
                top_spans.append(sp)

    for sp in _subtract((1, len(lines)), top_spans):
        text = _slice_lines(lines, sp)
        if _nonempty(text):
            out.append({"kind":"module","symbol":"module","span":sp,"text":text,"file_path":file_path,"chunk_index":idx}); idx += 1

    return out

# ----------------------------
# Editing helpers
# ----------------------------

def _line_to_abs_offsets(lines: List[str], span: Span) -> Tuple[int, int]:
    s, e = span
    s = max(1, s); e = max(s, e)
    start = sum(len(l) for l in lines[:s-1])
    end = sum(len(l) for l in lines[:e])  # exclusive
    return start, end

def _validate_python(text: str, filename: str = "<edited>") -> None:
    ast.parse(text, filename=filename)

def _resolve_symbol_node(tree: ast.Module, symbol: str) -> Optional[ast.AST]:
    parts = symbol.split(".")

    def find_in_scope(scope: List[ast.stmt], name: str) -> Optional[ast.AST]:
        for stmt in scope:
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if stmt.name == name:
                    return stmt
            elif isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        return stmt
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name) and stmt.target.id == name:
                    return stmt
        return None

    def walk(scope: List[ast.stmt], i: int) -> Optional[ast.AST]:
        if i >= len(parts): return None
        name = parts[i]
        node = find_in_scope(scope, name)

        if node is None:
            return None

        if i == len(parts) - 1:
            return node

        # Recurse if it's a container
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return walk(node.body, i+1)

        return None

    return walk(tree.body, 0)

def _docstring_span(src: str, node: ast.AST) -> Optional[Span]:
    body = getattr(node, "body", None)
    if not body: return None
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
        return _span_for(first, src, include_decorators=False)
    return None

def _first_body_linenum(node: ast.AST) -> Optional[int]:
    body = getattr(node, "body", None)
    if not body: return None
    return getattr(body[0], "lineno", None)

def _indent_for_body(lines: List[str], node: ast.AST) -> str:
    fn_line = getattr(node, "lineno", 1)
    body_line = _first_body_linenum(node)
    if body_line is None:
        body_line = fn_line + 1 if fn_line < len(lines) else len(lines)
    sample = lines[min(max(1, body_line) - 1, len(lines) - 1)]
    return sample[:len(sample) - len(sample.lstrip(" \t"))]

def _body_span(src: str, node: ast.AST, include_docstring: bool = True) -> Optional[Span]:
    body = getattr(node, "body", None)
    if not body: return None
    if include_docstring and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        start_node = body[0]
    else:
        start_node = body[0] if not include_docstring else body[0]  # already handled
        if include_docstring is False and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            if len(body) > 1:
                start_node = body[1]
    start = getattr(start_node, "lineno", getattr(node, "lineno", 1) + 1)
    end = _safe_get_end_lineno(src, body[-1])
    return (start, end)

def _indent_block(block: str, indent: str) -> str:
    if not block.endswith("\n"): block += "\n"
    return "".join((indent + line if line.strip() else line) for line in block.splitlines(keepends=True))

def _diff(a: str, b: str) -> str:
    return "".join(difflib.unified_diff(
        a.splitlines(keepends=True),
        b.splitlines(keepends=True),
        fromfile="before", tofile="after"
    ))

# ----------------------------
# Tool Definition
# ----------------------------

class EditorCommand(str, Enum):
    LIST = "list_symbols"
    SHOW = "show_symbol"
    REPLACE_WHOLE = "replace_whole"
    REPLACE_BODY = "replace_body"
    REPLACE_DOCSTRING = "replace_docstring"
    INSERT_AFTER = "insert_after"
    INSERT_BEFORE = "insert_before"
    DELETE = "delete_symbol"
    ADD_IMPORT = "add_import"
    REMOVE_IMPORT = "remove_import"

class ASTCodeEditorTool(BaseAnthropicTool):
    """
    AST-accurate Python code editor (no embeddings).
    Operates within REPO_DIR; targets symbols: module | func | Class | Class.method.
    """

    MAX_OUTPUT_LENGTH = 3000
    TRUNCATION_MARKER = " ... [TRUNCATED] ... "

    name: Literal["ast_code_editor"] = "ast_code_editor"
    api_type: Literal["custom"] = "custom"
    description: str = (
        "Edit Python source files via AST-perfect spans. "
        "Commands: list_symbols, show_symbol, replace_whole, replace_body, replace_docstring, insert_after, insert_before, delete_symbol, add_import, remove_import. "
        "All edits are validated by re-parsing before write; supports dry-run diffs."
    )

    def __init__(self, display=None):
        super().__init__(display=display)
        self.display = display

    def _truncate_string(self, s: str) -> str:
        if len(s) > self.MAX_OUTPUT_LENGTH:
            half = (self.MAX_OUTPUT_LENGTH - len(self.TRUNCATION_MARKER)) // 2
            return s[:half] + self.TRUNCATION_MARKER + s[-half:]
        return s

    # ---------- API schema ----------
    def to_params(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": [c.value for c in EditorCommand],
                            "description": "Command to execute"
                        },
                        "path": {
                            "type": "string",
                            "description": "Path to file, relative to REPO_DIR (Python file)"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Target symbol (e.g., 'module', 'foo', 'Class', 'Class.method')"
                        },
                        "text": {
                            "type": "string",
                            "description": "Inline text for edits (body/whole/docstring or inserted code)"
                        },
                        "from_file": {
                            "type": "string",
                            "description": "Path (relative to REPO_DIR) to read text for the edit"
                        },
                        "keep_docstring": {
                            "type": "boolean",
                            "default": True,
                            "description": "For replace_body: preserve existing docstring if present"
                        },
                        "dry_run": {
                            "type": "boolean",
                            "default": False,
                            "description": "Validate and show unified diff without writing changes"
                        },
                        "quote": {
                            "type": "string",
                            "enum": ['"""', "'''"],
                            "default": '"""',
                            "description": "Docstring quote style for replace_docstring"
                        },
                    },
                    "required": ["command"]
                }
            }
        }

    # ---------- Output formatting (match your house style) ----------
    def format_output(self, data: dict) -> str:
        lines = []
        lines.append(f"Command: {data.get('command')}")
        lines.append(f"Status: {data.get('status')}")

        # Optional common fields
        if "path" in data:
            lines.append(f"File: {data['path']}")
        if "symbol" in data and data["symbol"] is not None:
            lines.append(f"Symbol: {data['symbol']}")

        if data.get("status") == "error":
            lines.append("\nErrors:")
            lines.append(f"{data.get('error', 'Unknown error')}")

        if "symbols" in data:
            lines.append("\nSymbols:")
            for s in data["symbols"]:
                lines.append(f"- {s['kind']:7} {s['symbol']:30} lines {s['lines']}")

        if "show_text" in data:
            lines.append("\nOutput:")
            lines.append(self._truncate_string(data["show_text"]))

        if "diff" in data:
            lines.append("\nOutput:")
            # wrap diff in console block to match your pattern
            lines.append("```console")
            lines.extend(self._truncate_string(data["diff"]).splitlines())
            lines.append("```")

        if "message_lines" in data:
            lines.append("\nOutput:")
            lines.extend(data["message_lines"])

        return "\n".join(lines)

    def _emit_console(self, cwd: str, action: str, detail: str = "", path: Optional[str] = None, symbol: Optional[str] = None) -> None:
        if self.display is None:
            return

        command_line = f"$ {action}"
        if path:
            command_line += f" path='{path}'"
        if symbol:
            command_line += f" symbol='{symbol}'"

        block = ["```console"]
        block.append(f"$ cd {cwd}")
        block.append(command_line)
        if detail:
            block.extend(detail.rstrip().splitlines())
        block.append("```")
        self.display.add_message("assistant", "\n".join(block))

    # ---------- Symbol listing / lookup ----------
    def _list_symbols(self, repo: Path, rel_path: str) -> List[Dict]:
        file_path = (repo / rel_path).resolve()
        src = _pep263_read(file_path) or ""
        chunks = _iter_chunks_for_text(src, str(file_path))
        return [{
            "kind": c["kind"],
            "symbol": c["symbol"],
            "lines": f"{c['span'][0]}-{c['span'][1]}"
        } for c in chunks]

    def _show_symbol(self, repo: Path, rel_path: str, symbol: str) -> str:
        file_path = (repo / rel_path).resolve()
        src = _pep263_read(file_path) or ""
        chunks = _iter_chunks_for_text(src, str(file_path))
        cand = [c for c in chunks if c["symbol"] == symbol]
        if not cand:
            raise ValueError(f"Symbol not found: {symbol}")
        # Prefer most specific (method > function > class > module)
        kind_rank = {"method":3,"function":2,"class":1,"module":0}
        cand.sort(key=lambda c: (kind_rank.get(c["kind"], -1), c["span"][0], -c["span"][1]), reverse=True)
        return cand[0]["text"]

    # ---------- Core edit apply ----------
    def _apply(self, repo: Path, rel_path: str, mutate_fn, *, dry_run: bool) -> str:
        p = (repo / rel_path).resolve()
        src = _pep263_read(p)
        if src is None:
            raise FileNotFoundError(f"Cannot read file: {rel_path}")
        before = src
        after = mutate_fn(before, str(p))  # may raise on bad symbol or syntax
        if dry_run:
            diff = _diff(before, after)
            return diff
        tmp = p.with_suffix(p.suffix + f".tmp.{_sha1(after)[:8]}")
        tmp.write_text(after, encoding="utf-8")
        os.replace(tmp, p)
        return ""

    # ---------- Mutation helpers ----------
    def _replace_whole(self, src: str, filename: str, symbol: str, new_text: str) -> str:
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=filename)
        node = _resolve_symbol_node(tree, symbol)
        if node is None:
            raise ValueError(f"Symbol not found: {symbol}")
        span = _span_for(node, src, include_decorators=True)
        s_idx, e_idx = _line_to_abs_offsets(lines, span)
        result = src[:s_idx] + _ensure_trailing_nl(new_text) + src[e_idx:]
        _validate_python(result, filename)
        return result

    def _replace_docstring(self, src: str, filename: str, symbol: str, new_doc: str, triple: str) -> str:
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=filename)
        node = _resolve_symbol_node(tree, symbol)
        if node is None:
            raise ValueError(f"Symbol not found: {symbol}")

        doc_span = _docstring_span(src, node)
        indent = _indent_for_body(lines, node)
        doc_block = f'{indent}{triple}{new_doc}{triple}\n'

        if doc_span:
            s_idx, e_idx = _line_to_abs_offsets(lines, doc_span)
            result = src[:s_idx] + doc_block + src[e_idx:]
        else:
            insertion_line = _first_body_linenum(node) or (getattr(node, "lineno", 1) + 1)
            ins_idx = sum(len(l) for l in lines[:insertion_line-1])
            result = src[:ins_idx] + doc_block + src[ins_idx:]

        _validate_python(result, filename)
        return result

    def _replace_body(self, src: str, filename: str, symbol: str, new_body: str, keep_docstring: bool) -> str:
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=filename)
        node = _resolve_symbol_node(tree, symbol)
        if node is None:
            raise ValueError(f"Symbol not found: {symbol}")

        span = _body_span(src, node, include_docstring=keep_docstring)
        if span is None:
            # empty body: insert after header line
            start_line = getattr(node, "lineno", 1) + 1
            span = (start_line, start_line - 1)  # empty range

        indent = _indent_for_body(lines, node)
        new_block = _indent_block(new_body, indent)
        s_idx, e_idx = _line_to_abs_offsets(lines, span)
        result = src[:s_idx] + new_block + src[e_idx:]
        _validate_python(result, filename)
        return result

    def _delete_symbol(self, src: str, filename: str, symbol: str) -> str:
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=filename)
        node = _resolve_symbol_node(tree, symbol)
        if node is None:
            raise ValueError(f"Symbol not found: {symbol}")
        span = _span_for(node, src, include_decorators=True)
        s_idx, e_idx = _line_to_abs_offsets(lines, span)
        # eat following blank lines
        tail = src[e_idx:]
        rm_extra = 0
        for ch in tail.splitlines(keepends=True):
            if ch.strip():
                break
            rm_extra += len(ch)
        result = src[:s_idx] + src[e_idx + rm_extra:]
        _validate_python(result, filename)
        return result

    def _insert_after(self, src: str, filename: str, symbol: str, new_text: str) -> str:
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=filename)
        node = _resolve_symbol_node(tree, symbol)
        if node is None:
            raise ValueError(f"Symbol not found: {symbol}")
        span = _span_for(node, src, include_decorators=True)
        _, e_idx = _line_to_abs_offsets(lines, span)
        pad = "" if src[e_idx-1:e_idx] == "\n" else "\n"
        result = src[:e_idx] + pad + _ensure_trailing_nl(new_text) + src[e_idx:]
        _validate_python(result, filename)
        return result

    def _insert_before(self, src: str, filename: str, symbol: str, new_text: str) -> str:
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=filename)
        node = _resolve_symbol_node(tree, symbol)
        if node is None:
            raise ValueError(f"Symbol not found: {symbol}")
        span = _span_for(node, src, include_decorators=True)
        s_idx, _ = _line_to_abs_offsets(lines, span)
        result = src[:s_idx] + _ensure_trailing_nl(new_text) + src[s_idx:]
        _validate_python(result, filename)
        return result

    def _add_import(self, src: str, filename: str, new_import: str) -> str:
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=filename)
        
        # Check if already present (naive check)
        if new_import.strip() in src:
            # A better check would be AST based, but for now let's trust the user or do a simple string check
            pass

        last_import_end = 0
        has_imports = False
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                has_imports = True
                span = _span_for(node, src)
                _, e_idx = _line_to_abs_offsets(lines, span)
                last_import_end = max(last_import_end, e_idx)
        
        if has_imports:
            result = src[:last_import_end] + _ensure_trailing_nl(new_import) + src[last_import_end:]
        else:
            # Insert at top, after docstring if present
            insert_idx = 0
            if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
                span = _span_for(tree.body[0], src)
                _, e_idx = _line_to_abs_offsets(lines, span)
                insert_idx = e_idx
            
            result = src[:insert_idx] + _ensure_trailing_nl(new_import) + src[insert_idx:]
            
        _validate_python(result, filename)
        return result

    def _remove_import(self, src: str, filename: str, import_text: str) -> str:
        # This is tricky because import_text might be "import foo" or "from bar import baz"
        # We will look for exact line matches or AST matches. 
        # For simplicity, let's try to find the AST node that generates this source or matches the names.
        # Actually, simpler: find the node that *contains* the names in import_text?
        # Let's just do a best-effort AST match.
        
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src, filename=filename)
        
        target_node = None
        
        # Heuristic: parse the import_text to see what we are looking for
        try:
            imp_tree = ast.parse(import_text)
            if not imp_tree.body: return src
            imp_node = imp_tree.body[0]
        except:
            raise ValueError(f"Invalid import text: {import_text}")

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Compare node with imp_node
                if ast.dump(node) == ast.dump(imp_node):
                    target_node = node
                    break
        
        if target_node:
            span = _span_for(target_node, src)
            s_idx, e_idx = _line_to_abs_offsets(lines, span)
            # Remove trailing newline if present
            if e_idx < len(src) and src[e_idx] == '\n':
                e_idx += 1
            result = src[:s_idx] + src[e_idx:]
            _validate_python(result, filename)
            return result
            
        raise ValueError(f"Import not found: {import_text}")

    # ---------- Public entry (__call__) ----------
    async def __call__(  # type: ignore[override]
        self,
        *,
        command: EditorCommand | str,
        path: str | None = None,
        symbol: str | None = None,
        text: str | None = None,
        from_file: str | None = None,
        keep_docstring: bool = True,
        dry_run: bool = False,
        quote: str = '"""',
        **kwargs,
    ) -> ToolResult:
        try:
            # Normalize command input
            if hasattr(command, "value"):
                cmd_value = command.value  # Enum
            else:
                try:
                    command = EditorCommand(command)  # may raise
                    cmd_value = command.value
                except Exception:
                    return ToolResult(error=f"Unknown command: {command}",
                                      message=self.format_output({"command": command, "status": "error", "error": "Unknown command"}),
                                      tool_name=self.name)

            # Repo root
            repo_dir = get_constant("REPO_DIR")
            if not repo_dir:
                return ToolResult(error="REPO_DIR is not configured", tool_name=self.name)
            repo = Path(repo_dir)

            # Validate path for commands that need it
            needs_path = cmd_value in {
                EditorCommand.LIST.value, EditorCommand.SHOW.value,
                EditorCommand.REPLACE_WHOLE.value, EditorCommand.REPLACE_BODY.value,
                EditorCommand.REPLACE_DOCSTRING.value, EditorCommand.INSERT_AFTER.value,
                EditorCommand.INSERT_BEFORE.value, EditorCommand.DELETE.value,
                EditorCommand.ADD_IMPORT.value, EditorCommand.REMOVE_IMPORT.value
            }
            if needs_path and not path:
                return ToolResult(error="Missing 'path'", message=self.format_output({"command": cmd_value, "status": "error", "error": "Missing 'path'"}), tool_name=self.name)

            if path:
                abs_path = (repo / path).resolve()
                if not _within_repo(repo, abs_path):
                    return ToolResult(error="Path escapes REPO_DIR", message=self.format_output({"command": cmd_value, "status": "error", "error": "Path escapes REPO_DIR", "path": path}), tool_name=self.name)
                if not abs_path.exists():
                    return ToolResult(error="File not found", message=self.format_output({"command": cmd_value, "status": "error", "error": "File not found", "path": path}), tool_name=self.name)

            # Read optional edit text (inline or from_file)
            def _load_edit_text() -> Optional[str]:
                if text is not None:
                    return text
                if from_file:
                    srcp = (repo / from_file).resolve()
                    if not _within_repo(repo, srcp):
                        raise ValueError("from_file escapes REPO_DIR")
                    data = _pep263_read(srcp)
                    if data is None:
                        raise ValueError(f"Unable to read from_file: {from_file}")
                    return data
                return None

            # Dispatch
            if cmd_value == EditorCommand.LIST.value:
                symbols = self._list_symbols(repo, path)
                payload = {"command": cmd_value, "status": "success", "path": path, "symbols": symbols}
                output_str = "\n".join(f"{s['kind']:7} {s['symbol']:30} lines {s['lines']}" for s in symbols)
                self._emit_console(str(repo), f"{self.name} {cmd_value}", detail=output_str, path=path)
                return ToolResult(output=output_str,
                                  message=self.format_output(payload),
                                  command=cmd_value, tool_name=self.name)

            if cmd_value == EditorCommand.SHOW.value:
                if not symbol:
                    return ToolResult(error="Missing 'symbol'", message=self.format_output({"command": cmd_value, "status":"error","error":"Missing 'symbol'","path":path}), tool_name=self.name)
                show_text = self._show_symbol(repo, path, symbol)
                payload = {"command": cmd_value, "status": "success", "path": path, "symbol": symbol, "show_text": show_text}
                self._emit_console(str(repo), f"{self.name} {cmd_value}", detail=show_text, path=path, symbol=symbol)
                return ToolResult(output=show_text, message=self.format_output(payload), command=cmd_value, tool_name=self.name)

            # Mutations
            if cmd_value in {
                EditorCommand.REPLACE_WHOLE.value, EditorCommand.REPLACE_BODY.value,
                EditorCommand.REPLACE_DOCSTRING.value, EditorCommand.INSERT_AFTER.value,
                EditorCommand.INSERT_BEFORE.value,
                EditorCommand.DELETE.value
            } and symbol is None and cmd_value != EditorCommand.DELETE.value:
                return ToolResult(error="Missing 'symbol'", message=self.format_output({"command": cmd_value, "status":"error","error":"Missing 'symbol'","path":path}), tool_name=self.name)

            # Load edit content if needed
            edit_text: Optional[str] = None
            if cmd_value in {
                EditorCommand.REPLACE_WHOLE.value,
                EditorCommand.REPLACE_BODY.value,
                EditorCommand.REPLACE_DOCSTRING.value,
                EditorCommand.INSERT_AFTER.value,
                EditorCommand.INSERT_BEFORE.value,
                EditorCommand.ADD_IMPORT.value,
                EditorCommand.REMOVE_IMPORT.value
            }:
                edit_text = _load_edit_text()
                if edit_text is None:
                    return ToolResult(error="Missing 'text' or 'from_file'", message=self.format_output({"command": cmd_value, "status":"error","error":"Provide 'text' or 'from_file'","path":path,"symbol":symbol}), tool_name=self.name)

            # Perform operation
            diff_text = ""
            if cmd_value == EditorCommand.REPLACE_WHOLE.value:
                def mutate(before: str, filename: str) -> str:
                    return self._replace_whole(before, filename, symbol, edit_text)
                diff_text = self._apply(repo, path, mutate, dry_run=dry_run)

            elif cmd_value == EditorCommand.REPLACE_BODY.value:
                def mutate(before: str, filename: str) -> str:
                    return self._replace_body(before, filename, symbol, edit_text, keep_docstring)
                diff_text = self._apply(repo, path, mutate, dry_run=dry_run)

            elif cmd_value == EditorCommand.REPLACE_DOCSTRING.value:
                def mutate(before: str, filename: str) -> str:
                    return self._replace_docstring(before, filename, symbol, edit_text.rstrip("\n"), quote)
                diff_text = self._apply(repo, path, mutate, dry_run=dry_run)

            elif cmd_value == EditorCommand.INSERT_AFTER.value:
                def mutate(before: str, filename: str) -> str:
                    return self._insert_after(before, filename, symbol, edit_text)
                diff_text = self._apply(repo, path, mutate, dry_run=dry_run)

            elif cmd_value == EditorCommand.INSERT_BEFORE.value:
                def mutate(before: str, filename: str) -> str:
                    return self._insert_before(before, filename, symbol, edit_text)
                diff_text = self._apply(repo, path, mutate, dry_run=dry_run)

            elif cmd_value == EditorCommand.ADD_IMPORT.value:
                def mutate(before: str, filename: str) -> str:
                    return self._add_import(before, filename, edit_text)
                diff_text = self._apply(repo, path, mutate, dry_run=dry_run)

            elif cmd_value == EditorCommand.REMOVE_IMPORT.value:
                def mutate(before: str, filename: str) -> str:
                    return self._remove_import(before, filename, edit_text)
                diff_text = self._apply(repo, path, mutate, dry_run=dry_run)

            elif cmd_value == EditorCommand.DELETE.value:
                if symbol is None:
                    return ToolResult(error="Missing 'symbol'", message=self.format_output({"command": cmd_value, "status":"error","error":"Missing 'symbol'","path":path}), tool_name=self.name)
                def mutate(before: str, filename: str) -> str:
                    return self._delete_symbol(before, filename, symbol)
                diff_text = self._apply(repo, path, mutate, dry_run=dry_run)

            # Emit console-like block to display (consistent with your tool)
            self._emit_console(str(repo), f"{self.name} {cmd_value}", detail=(diff_text if dry_run and diff_text else ""), path=path, symbol=symbol)

            payload = {
                "command": cmd_value,
                "status": "success",
                "path": path,
                "symbol": symbol,
            }
            if dry_run:
                payload["diff"] = diff_text

            msg = self.format_output(payload)
            return ToolResult(output=diff_text if dry_run else "OK",
                              message=msg, command=cmd_value, tool_name=self.name)

        except Exception as e:
            err = str(e)
            payload = {"command": command if isinstance(command, str) else command.value,
                       "status": "error", "error": err, "path": path, "symbol": symbol}
            return ToolResult(error=err, message=self.format_output(payload), command=str(command), tool_name=self.name)
