"""
edit_tool_robust.py — v7  (complete, self‑contained)
====================================================
Cross‑platform file editor geared for **code refactoring & debugging**.

Commands
--------
view • create • str_replace • insert • undo_edit • lint • format

Highlights
~~~~~~~~~~
* **Code‑aware replacement** (`str_replace`) with three modes:
  • exact  (default) – literal match.
  • regex            – `old_str` interpreted as a Python regex.
  • fuzzy            – whitespace‑agnostic pattern built automatically.

* Atomic, lock‑safe writes with `portalocker`, size‑guarded reads
  (≤ 512 KiB per file).

* Built‑in lint (`ruff`→`pylint`) and formatter (`black`).

* Strict path sandboxing: files stay under `REPO_DIR`.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import difflib

import portalocker  # type: ignore

from .base import BaseTool, ToolError, ToolResult
from config import get_constant
from utils.file_logger import log_file_operation
from utils.llm_client import create_llm_client

# ---------------------------------------------------------------------------
# Configuration & constants
# ---------------------------------------------------------------------------

_LOG = logging.getLogger(__name__)
MAX_FILE_BYTES = 512 * 1024  # 512 KiB file size cap
MAX_STDOUT = 10_000  # cap subprocess output in chars
SNIPPET_LINES = 4  # context lines around edits

# ---------------------------------------------------------------------------
# Command enumeration
# ---------------------------------------------------------------------------


class Command(Enum):
    VIEW = "view"
    CREATE = "create"
    STR_REPLACE = "str_replace"
    INSERT = "insert"
    UNDO_EDIT = "undo_edit"
    LINT = "lint"
    FORMAT = "format"

    @classmethod
    def list(cls) -> List[str]:
        return [c.value for c in cls]


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


class EditTool(BaseTool):
    """Cross‑platform editor designed for codebases."""

    api_type = "text_editor_20250124"

    @property
    def name(self) -> str:  # satisfies BaseTool contract
        return "str_replace_editor"

    @property
    def description(self) -> str:  # satisfies BaseTool contract
        return (
            "A cross‑platform file editor that can view, create, edit, lint, and "
            "format files inside the project sandbox. Supports regex / fuzzy code "
            "replacements for safer refactors."
        )

    # ------------------------------------------------------------------
    # Construction / schema
    # ------------------------------------------------------------------

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display)
        self.display = display
        self._repo_dir: Path = Path(get_constant("REPO_DIR")).resolve()
        self._file_history: Dict[Path, List[str]] = defaultdict(list)

    def to_params(self) -> Dict[str, Any]:
        """Expose the OpenAI function‑calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "enum": Command.list()},
                        "path": {"type": "string"},
                        "file_text": {"type": "string"},
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "insert_line": {"type": "integer"},
                        "match_mode": {
                            "type": "string",
                            "enum": ["exact", "regex", "fuzzy"],
                            "description": "Replacement strategy for str_replace (default exact).",
                        },
                        "use_llm": {
                            "type": "boolean",
                            "description": "If true, perform the edit via LLM and save the returned full file",
                        },
                        "llm_model": {
                            "type": "string",
                            "description": "Override model for LLM editing (defaults to EDIT_LLM_MODEL or MAIN_MODEL)",
                        },
                    },
                    "required": ["command", "path"],
                },
            },
        }

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    async def __call__(
        self,
        *,
        command: str,
        path: str,
        file_text: Optional[str] = None,
        view_range: Optional[List[int]] = None,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        insert_line: Optional[int] = None,
        match_mode: str = "exact",
        use_llm: Optional[bool] = None,
        llm_model: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        cmd_enum = self._validate_command(command)
        _path = self._resolve_path(path)
        call_args = {
            "command": cmd_enum.value,
            "path": str(_path),
            "match_mode": match_mode,
            "old_str": old_str,
            "new_str": new_str,
            "insert_line": insert_line,
            "use_llm": use_llm,
        }
        try:
            if cmd_enum is Command.CREATE:
                result = self._cmd_create(_path, file_text)
            elif cmd_enum is Command.VIEW:
                result = await self._cmd_view(_path, view_range)
            elif cmd_enum is Command.STR_REPLACE:
                effective_use_llm = (
                    bool(use_llm)
                    if use_llm is not None
                    else bool(get_constant("EDIT_WITH_LLM", False))
                )
                if effective_use_llm:
                    result = await self._cmd_str_replace_llm(_path, old_str, new_str, match_mode, llm_model)
                else:
                    result = self._cmd_str_replace(_path, old_str, new_str, match_mode)
            elif cmd_enum is Command.INSERT:
                effective_use_llm = (
                    bool(use_llm)
                    if use_llm is not None
                    else bool(get_constant("EDIT_WITH_LLM", False))
                )
                if effective_use_llm:
                    result = await self._cmd_insert_llm(_path, insert_line, new_str, llm_model)
                else:
                    result = self._cmd_insert(_path, insert_line, new_str)
            elif cmd_enum is Command.UNDO_EDIT:
                result = self._cmd_undo(_path)
            elif cmd_enum is Command.LINT:
                result = self._cmd_lint(_path)
            elif cmd_enum is Command.FORMAT:
                result = self._cmd_format(_path)
            else:
                raise ToolError(f"Unhandled command {cmd_enum}")

            # Add assistant console-style display if configured
            self._maybe_display(cmd_enum, _path, result, call_args=call_args)
            return result
        except Exception as exc:
            _LOG.error("EditTool failure", exc_info=True)
            error_result = ToolResult(
                output=f"EditTool error running {command} on {_path}: {exc}",
                error=str(exc),
                tool_name=self.name,
                command=command,
            )
            self._maybe_display(cmd_enum, _path, error_result, is_error=True, call_args=call_args)
            return error_result

    # ------------------------------------------------------------------
    # Command validation
    # ------------------------------------------------------------------

    def _validate_command(self, cmd: str) -> Command:
        try:
            return Command(cmd)
        except ValueError:
            raise ToolError(
                f"Invalid command '{cmd}'. Valid commands: {', '.join(Command.list())}"
            )

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _cmd_create(self, path: Path, content: Optional[str]) -> ToolResult:
        if content is None:
            raise ToolError("`file_text` required for create")
        self._write_file(path, content)
        return ToolResult(output=f"Created {path}")

    async def _cmd_view(self, path: Path, vrange: Optional[List[int]]) -> ToolResult:
        if path.is_dir():
            return ToolResult(output=self._dir_list(path))
        if path.is_file():
            return self._file_view(path, vrange)
        raise ToolError(f"{path} is neither file nor directory")

    def _cmd_str_replace(
        self, path: Path, old: Optional[str], new: Optional[str], mode: str
    ) -> ToolResult:
        if old is None:
            raise ToolError("`old_str` required for str_replace")
        if mode not in {"exact", "regex", "fuzzy"}:
            mode = "exact"

        try:
            return self._file_str_replace(path, old, new or "", mode)
        except ToolError as err:
            # If exact/regex match fails to find a match, automatically retry with fuzzy matching
            message = str(err)
            if mode in {"exact", "regex"} and ("No match" in message or "No match found" in message):
                fallback_result = self._file_str_replace(path, old, new or "", "fuzzy")
                # Prepend a brief note so the user understands why it succeeded
                if fallback_result.output:
                    fallback_result.output = (
                        "[fallback to fuzzy match]\n" + fallback_result.output
                    )
                return fallback_result
            raise

    def _cmd_insert(
        self, path: Path, line: Optional[int], text: Optional[str]
    ) -> ToolResult:
        if line is None or text is None:
            raise ToolError("insert_line and new_str required for insert")
        return self._file_insert(path, line, text)

    async def _cmd_insert_llm(
        self, path: Path, line: Optional[int], text: Optional[str], model: Optional[str]
    ) -> ToolResult:
        if line is None or text is None:
            raise ToolError("insert_line and new_str required for insert")
        current = self._read_file(path)
        instructions = (
            "Insert the provided text at the specified line number. "
            "Preserve all other content and formatting exactly."
        )
        request_details = (
            f"Operation: insert\n"
            f"File: {self._rel_path(path)}\n"
            f"Insert line (0-based index means before line 1 is 0): {line}\n"
            f"Inserted text follows between <<<INSERT>>> markers.\n"
            f"<<<INSERT>>>\n{text}\n<<<INSERT>>>\n"
        )
        updated = await self._call_llm_full_file(
            model=model,
            path=path,
            instructions=instructions,
            request_details=request_details,
            current_content=current,
        )
        normalized = self._normalize_to_original_newlines(original=current, modified=updated)
        self._write_file(path, normalized)
        snippet = self._diff_snippet(current, normalized)
        return ToolResult(output=f"Inserted text in {path} via LLM\n{snippet}")

    def _cmd_undo(self, path: Path) -> ToolResult:
        return self._file_undo(path)

    def _cmd_lint(self, path: Path) -> ToolResult:
        runner = ["ruff", "--quiet", "--format", "text", str(path)]
        if not _which("ruff"):
            runner = ["python", "-m", "pylint", "--score", "n", str(path)]
        out, err, code = self._run(runner, cwd=self._repo_dir)
        tool = "ruff" if "ruff" in runner[0] else "pylint"
        header = f"{tool} exit {code}"
        return ToolResult(output=f"{header}\n{_truncate(out or err)}")

    def _cmd_format(self, path: Path) -> ToolResult:
        if not _which("black"):
            raise ToolError("Black not installed in environment")
        runner = ["black", "--quiet", str(path)]
        _, err, code = self._run(runner, cwd=self._repo_dir)
        status = "formatted" if code == 0 else "failed"
        return ToolResult(output=f"black {status}: {path}\n{err or ''}")

    # ------------------------------------------------------------------
    # Display helpers (console-style formatting similar to bash/write_code)
    # ------------------------------------------------------------------

    def _maybe_display(
        self,
        cmd_enum: Command,
        path: Path,
        result: ToolResult,
        *,
        is_error: bool = False,
        call_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.display is None:
            return
        try:
            formatted = self._format_terminal_output(cmd_enum, path, result, is_error=is_error, call_args=call_args)
            if formatted:
                # send as assistant message so it appears in console like other tools
                self.display.add_message("assistant", formatted)
        except Exception:  # pragma: no cover - display issues shouldn't break tool
            pass

    def _format_terminal_output(
        self,
        cmd_enum: Command,
        path: Path,
        result: ToolResult,
        *,
        is_error: bool = False,
        max_lines: int = 20,
        max_chars: int = 2_000,
        call_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return a fenced console block showing the invoked edit command and its output.

        Mirrors style: ```console + $ command + stdout/stderr lines + ```
        Truncates excessive output similar to other tools.
        """
        try:
            rel_path: str
            try:
                rel_path = str(path.relative_to(self._repo_dir))
            except Exception:
                rel_path = str(path)
            mode_arg = ""
            if call_args and cmd_enum is Command.STR_REPLACE:
                mm = call_args.get("match_mode") or "exact"
                mode_arg = f" --mode={mm}"
            invoked = f"$ edit {cmd_enum.value} {rel_path}{mode_arg}".rstrip()
            lines: List[str] = ["```console", invoked]

            # For str_replace, include previews of old/new strings attempted
            if call_args and cmd_enum is Command.STR_REPLACE:
                def _preview(label: str, value: Optional[str]) -> List[str]:
                    if value is None:
                        return [f"{label}: <none>"]
                    text = value
                    LIMIT = 400
                    if len(text) > LIMIT:
                        text = text[: LIMIT // 2] + "\n… (truncated) …\n" + text[-LIMIT // 2 :]
                    return [
                        f"{label} (len={len(value)}):",
                        "<<<",
                        text,
                        ">>>",
                    ]

                lines.extend(_preview("old_str", call_args.get("old_str")))
                lines.extend(_preview("new_str", call_args.get("new_str")))
            out_text = result.output or ""
            err_text = result.error or ""

            # If output already contains multiple lines, preserve them; else single line ok
            combined = out_text
            if err_text and err_text not in combined:
                if combined:
                    combined = combined.rstrip() + ("\n" if not combined.endswith("\n") else "") + err_text
                else:
                    combined = err_text

            if is_error and not err_text and not out_text:
                combined = "Error occurred but no output captured"

            # Truncate by chars first
            truncated = False
            if len(combined) > max_chars:
                combined = (
                    combined[: max_chars // 2]
                    + "\n… (truncated) …\n"
                    + combined[-max_chars // 2 :]
                )
                truncated = True
            # Split to lines and limit
            original_lines = combined.splitlines()
            if len(original_lines) > max_lines:
                head = original_lines[: max_lines // 2]
                tail = original_lines[-max_lines // 2 :]
                omitted = len(original_lines) - len(head) - len(tail)
                combined_lines = head + [f"… ({omitted} lines omitted) …"] + tail
                truncated = True
            else:
                combined_lines = original_lines
            lines.extend(combined_lines)
            if truncated:
                lines.append("[output truncated]")
            lines.append("```")
            return "\n".join(lines)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, p: str | Path) -> Path:
        p = Path(p)
        if not p.is_absolute():
            p = self._repo_dir / p
        p = p.resolve()
        if not str(p).startswith(str(self._repo_dir)):
            raise ToolError("Path escapes REPO_DIR sandbox")
        return p

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _read_file(self, path: Path) -> str:
        data = path.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise ToolError("File too large to load")
        return data.decode("utf-8", errors="replace")

    def _write_file(self, path: Path, content: str):
        if len(content.encode()) > MAX_FILE_BYTES:
            raise ToolError("Refusing to write >512 KiB file")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        # Write bytes to avoid platform newline translation (prevents accidental blank lines)
        with portalocker.Lock(str(tmp), "wb", timeout=5) as fp:
            fp.write(content.encode("utf-8"))
        # Windows‑safe timestamp (no ':')
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak.{timestamp}")
        if path.exists():
            shutil.copy2(path, backup)
            self._file_history[path].append(self._read_file(path))
        shutil.move(tmp, path)
        log_file_operation(path, "modify")

    # ------------------------------------------------------------------
    # Directory listing
    # ------------------------------------------------------------------

    def _dir_list(self, root: Path) -> str:
        entries: List[str] = []
        for depth in range(3):
            glob = os.path.join(*["*"] * (depth + 1)) if depth else "*"
            for item in root.glob(glob):
                if any(part.startswith(".") for part in item.parts):
                    continue
                entries.append(str(item.relative_to(self._repo_dir)))
        return "\n".join(sorted(entries))

    # ------------------------------------------------------------------
    # View file
    # ------------------------------------------------------------------

    def _file_view(self, path: Path, vrange: Optional[List[int]]) -> ToolResult:
        text = self._read_file(path)
        lines = text.splitlines()
        start = 1
        end = len(lines)
        if vrange:
            if len(vrange) != 2 or not all(isinstance(i, int) for i in vrange):
                raise ToolError("view_range must be [start, end]")
            start, end = vrange
            if start < 1 or end < -1:
                raise ToolError("Invalid view_range values")
            end = len(lines) if end == -1 else end
        snippet = "\n".join(self._numbered(lines[start - 1 : end], offset=start))
        return ToolResult(output=snippet)

    # ------------------------------------------------------------------
    # Replace string
    # ------------------------------------------------------------------

    def _file_str_replace(
        self, path: Path, old: str, new: str, mode: str
    ) -> ToolResult:
        text = self._read_file(path)
        pattern: str
        flags = re.DOTALL
        if mode == "exact":
            # Escape all characters, then allow CRLF/LF equivalence by turning literal newlines into \r?\n
            pattern = re.escape(old)
            pattern = self._crlf_tolerant(pattern)
        elif mode == "regex":
            # Respect user regex but make line breaks CRLF/LF tolerant
            pattern = self._crlf_tolerant(old)
        else:  # fuzzy
            pattern = self._build_fuzzy_regex(old)
        matches = list(re.finditer(pattern, text, flags))
        if not matches:
            raise ToolError("No match found for replacement")
        if len(matches) > 1:
            # Report line numbers of all matches to help caller refine pattern
            line_numbers: List[int] = []
            for m in matches:
                line_numbers.append(text.count("\n", 0, m.start()) + 1)
            raise ToolError(
                "Multiple matches found; make pattern more specific. Matches at lines: "
                + ", ".join(str(n) for n in line_numbers)
            )
        m = matches[0]
        candidate_text = text[: m.start()] + new + text[m.end() :]
        # Normalize EOLs to match original file style and trailing newline policy
        normalized_text = self._normalize_to_original_newlines(original=text, modified=candidate_text)
        self._write_file(path, normalized_text)
        # create context diff
        snippet = self._snippet(normalized_text, m.start(), m.start() + len(new))
        return ToolResult(output=f"Replaced code in {path}\n{snippet}")

    async def _cmd_str_replace_llm(
        self, path: Path, old: Optional[str], new: Optional[str], mode: str, model: Optional[str]
    ) -> ToolResult:
        if old is None:
            raise ToolError("`old_str` required for str_replace")
        new_text = new or ""
        current = self._read_file(path)
        if mode not in {"exact", "regex", "fuzzy"}:
            mode = "exact"

        instructions = (
            "Replace text in the file according to the details provided. "
            "Keep all unrelated content identical. Preserve indentation characters (tabs vs spaces), widths, and avoid reformatting."
        )
        request_details = (
            f"Operation: str_replace\n"
            f"File: {self._rel_path(path)}\n"
            f"Match mode: {mode}\n"
            f"Old snippet follows between <<<OLD>>> markers.\n"
            f"<<<OLD>>>\n{old}\n<<<OLD>>>\n"
            f"New snippet follows between <<<NEW>>> markers.\n"
            f"<<<NEW>>>\n{new_text}\n<<<NEW>>>\n"
            "Apply the replacement to the single correct location in the file."
        )

        updated = await self._call_llm_full_file(
            model=model,
            path=path,
            instructions=instructions,
            request_details=request_details,
            current_content=current,
        )
        normalized = self._normalize_to_original_newlines(original=current, modified=updated)
        self._write_file(path, normalized)
        snippet = self._diff_snippet(current, normalized)
        return ToolResult(output=f"Replaced code in {path} via LLM\n{snippet}")

    def _build_fuzzy_regex(self, s: str) -> str:
        """Construct a regex that matches the given string with flexible whitespace.

        - All non-whitespace characters are escaped literally
        - Any run of whitespace (spaces, tabs, newlines) becomes \\s+
        This avoids accidental regex metacharacter greediness that can span many lines.
        """
        parts = re.findall(r"\s+|\S+", s)
        escaped_segments: List[str] = []
        for part in parts:
            if part.isspace():
                escaped_segments.append(r"\s+")
            else:
                escaped_segments.append(re.escape(part))
        pattern = "".join(escaped_segments)
        # Coalesce any accidental consecutive \s+ tokens
        pattern = re.sub(r"(?:\\s\+){2,}", r"\\s+", pattern)
        return pattern

    def _crlf_tolerant(self, pattern: str) -> str:
        """Convert literal newlines in a pattern to a regex that matches either LF or CRLF.

        This replaces actual newline characters in the pattern with the regex sequence
        \\r?\\n so the same pattern will match files regardless of line ending style.
        """
        # First normalize any explicit CRLF in the pattern, then standalone LF
        pattern = pattern.replace("\r\n", r"\r?\n")
        pattern = pattern.replace("\n", r"\r?\n")
        return pattern

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def _file_insert(self, path: Path, line: int, text: str) -> ToolResult:
        if line < 0:
            raise ToolError("insert_line must be non‑negative")
        current = self._read_file(path)
        lines = current.splitlines()
        if line > len(lines):
            raise ToolError("insert_line beyond EOF")
        inserted_lines = text.splitlines()
        candidate_content = "\n".join(lines[:line] + inserted_lines + lines[line:])
        # Normalize to original file's newline convention and trailing newline policy
        new_content = self._normalize_to_original_newlines(original=current, modified=candidate_content)
        self._write_file(path, new_content)
        new_lines_all = new_content.splitlines()
        snippet_start = max(0, line - SNIPPET_LINES)
        snippet_end = line + SNIPPET_LINES + len(inserted_lines)
        snippet = "\n".join(
            self._numbered(
                new_lines_all[snippet_start:snippet_end],
                offset=max(1, snippet_start + 1),
            )
        )
        return ToolResult(output=f"Inserted text in {path}\n{snippet}")

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def _file_undo(self, path: Path) -> ToolResult:
        if not self._file_history[path]:
            raise ToolError("No edits to undo for this file")
        prev = self._file_history[path].pop()
        self._write_file(path, prev)
        return ToolResult(output=f"Undo successful for {path}")

    # ------------------------------------------------------------------
    # Helpers: numbering, snippet, runner, etc.
    # ------------------------------------------------------------------

    def _numbered(self, lines: List[str], *, offset: int = 1) -> List[str]:
        return [f"{idx + offset:6}\t{line}" for idx, line in enumerate(lines)]

    def _snippet(self, text: str, start: int, end: int) -> str:
        # Compute line numbers around the replaced region
        pre = text[:start]
        line_start = pre.count("\n")
        first = max(0, line_start - SNIPPET_LINES)
        all_lines = text.splitlines()
        last = min(len(all_lines), line_start + SNIPPET_LINES + 1)
        numbered = self._numbered(all_lines[first:last], offset=first + 1)
        return "\n".join(numbered)

    def _run(self, cmd: List[str], *, cwd: Path) -> Tuple[str, str, int]:
        proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=15)
        out = proc.stdout[:MAX_STDOUT]
        err = proc.stderr[:MAX_STDOUT]
        return out, err, proc.returncode

    # ------------------------------------------------------------------
    # Newline normalization helpers
    # ------------------------------------------------------------------

    def _detect_original_newline(self, text: str) -> Tuple[str, bool]:
        """Detect the predominant newline sequence and trailing newline policy in original text.

        Returns a tuple of (newline_sequence, has_trailing_newline).
        newline_sequence is "\r\n" if the file consistently uses CRLF, otherwise "\n".
        """
        uses_crlf = "\r\n" in text
        stray_lf_present = "\n" in text.replace("\r\n", "")
        newline_seq = "\r\n" if uses_crlf and not stray_lf_present else "\n"
        trailing = text.endswith(("\r\n", "\n"))
        return newline_seq, trailing

    def _normalize_to_original_newlines(self, *, original: str, modified: str) -> str:
        """Normalize newline characters in modified text to match the original file.

        - Preserves the original file's newline sequence (LF vs CRLF)
        - Preserves whether the file ended with a trailing newline
        - Avoids introducing extra blank lines at EOF
        """
        newline_seq, trailing = self._detect_original_newline(original)

        logical_lines = modified.splitlines()
        normalized = newline_seq.join(logical_lines)
        if trailing:
            if not normalized.endswith(newline_seq):
                normalized += newline_seq
        else:
            if normalized.endswith("\r\n"):
                normalized = normalized[:-2]
            elif normalized.endswith("\n"):
                normalized = normalized[:-1]
        return normalized

    # ------------------------------------------------------------------
    # LLM edit helpers
    # ------------------------------------------------------------------

    async def _call_llm_full_file(
        self,
        *,
        model: Optional[str],
        path: Path,
        instructions: str,
        request_details: str,
        current_content: str,
    ) -> str:
        """Call an LLM with a prompt that includes full file content and returns full updated file.

        The LLM is instructed to return ONLY the updated file content (no fences or explanations).
        """
        selected_model = model or get_constant("EDIT_LLM_MODEL", None) or get_constant("MAIN_MODEL", "anthropic/claude-sonnet-4")
        client = create_llm_client(str(selected_model))

        rel = self._rel_path(path)
        system_msg = (
            "You are a code editing engine. Apply the requested change precisely to the provided file. "
            "Return ONLY the complete updated file content. Do not include code fences, diffs, or explanations. "
            "Preserve original indentation characters (tabs vs spaces) and indentation widths. "
            "Preserve all unaffected lines exactly."
        )
        user_msg = (
            f"Instructions:\n{instructions}\n\n"
            f"Details:\n{request_details}\n\n"
            f"Current file ({rel}) content between <<<FILE>>> markers follows. Modify it and return the entire updated file with no extra text.\n"
            f"<<<FILE>>>\n{current_content}\n<<<FILE>>>\n"
            "Output policy: Return only the updated file content, nothing else."
        )

        # Build messages in OpenAI-style format
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Use generous max tokens since file can be large; client will enforce provider limits.
        llm_response = await client.call(messages=messages, max_tokens=200000, temperature=0.0)
        cleaned = self._extract_updated_content(llm_response)
        if not cleaned:
            raise ToolError("LLM returned empty content for updated file")
        return cleaned

    def _extract_updated_content(self, text: str) -> str:
        """Extract the updated file content from an LLM response.

        If the model included code fences or markers despite instructions, extract the most plausible block.
        """
        stripped = text.strip()
        # If fenced block exists, prefer the largest fenced block
        fence_matches = list(re.finditer(r"```[a-zA-Z0-9_+-]*\n([\s\S]*?)\n```", stripped))
        if fence_matches:
            largest = max(fence_matches, key=lambda m: len(m.group(1)))
            return largest.group(1)
        # If markers exist
        m = re.search(r"<<<FILE>>>\n([\s\S]*?)\n<<<FILE>>>", stripped)
        if m:
            return m.group(1)
        # Otherwise return as-is
        return stripped

    def _diff_snippet(self, before: str, after: str, context: int = SNIPPET_LINES) -> str:
        """Produce a compact unified diff snippet for display."""
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        diff_lines = list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile="before",
                tofile="after",
                n=context,
                lineterm="",
            )
        )
        # Limit diff to a reasonable length
        MAX_DIFF_LINES = 200
        if len(diff_lines) > MAX_DIFF_LINES:
            head = diff_lines[: MAX_DIFF_LINES // 2]
            tail = diff_lines[-MAX_DIFF_LINES // 2 :]
            diff_lines = head + ["… (diff truncated) …"] + tail
        return "\n".join(diff_lines)

    def _rel_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._repo_dir))
        except Exception:
            return str(path)


# ---------------------------------------------------------------------------
# Module‑level helpers
# ---------------------------------------------------------------------------


def _which(exe: str) -> Optional[str]:
    """Return path to executable or None."""
    return shutil.which(exe)


def _truncate(text: str, max_len: int = MAX_STDOUT) -> str:
    """Truncate long strings and indicate omission."""
    return text if len(text) <= max_len else text[:max_len] + "\n⋯ (truncated)"
