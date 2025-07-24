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
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import portalocker  # type: ignore

from .base import BaseTool, ToolError, ToolResult
from config import get_constant
from utils.file_logger import log_file_operation

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
    name = "str_replace_editor"

    description = (
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
        **kwargs,
    ) -> ToolResult:
        cmd_enum = self._validate_command(command)
        _path = self._resolve_path(path)
        try:
            if cmd_enum is Command.CREATE:
                return self._cmd_create(_path, file_text)
            if cmd_enum is Command.VIEW:
                return await self._cmd_view(_path, view_range)
            if cmd_enum is Command.STR_REPLACE:
                return self._cmd_str_replace(_path, old_str, new_str, match_mode)
            if cmd_enum is Command.INSERT:
                return self._cmd_insert(_path, insert_line, new_str)
            if cmd_enum is Command.UNDO_EDIT:
                return self._cmd_undo(_path)
            if cmd_enum is Command.LINT:
                return self._cmd_lint(_path)
            if cmd_enum is Command.FORMAT:
                return self._cmd_format(_path)
            raise ToolError(f"Unhandled command {cmd_enum}")
        except Exception as exc:
            _LOG.error("EditTool failure", exc_info=True)
            return ToolResult(
                output=f"EditTool error running {command} on {_path}: {exc}",
                error=str(exc),
                tool_name=self.name,
                command=command,
            )

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
            raise ToolError("match_mode must be exact/regex/fuzzy")
        return self._file_str_replace(path, old, new or "", mode)

    def _cmd_insert(
        self, path: Path, line: Optional[int], text: Optional[str]
    ) -> ToolResult:
        if line is None or text is None:
            raise ToolError("insert_line and new_str required for insert")
        return self._file_insert(path, line, text)

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
            raise ToolError("Refusing to write >512 KiB file")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(tmp), "w", timeout=5) as fp:
            fp.write(content)
        # Windows‑safe timestamp (no ':')
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
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
            pattern = re.escape(old)
        elif mode == "regex":
            pattern = old
        else:  # fuzzy
            pattern = self._build_fuzzy_regex(old)
        matches = list(re.finditer(pattern, text, flags))
        if not matches:
            raise ToolError("No match found for replacement")
        if len(matches) > 1:
            raise ToolError("Multiple matches found; make pattern more specific")
        m = matches[0]
        new_text = text[: m.start()] + new + text[m.end() :]
        self._write_file(path, new_text)
        # create context diff
        snippet = self._snippet(new_text, m.start(), m.start() + len(new))
        return ToolResult(output=f"Replaced code in {path}\n{snippet}")

    def _build_fuzzy_regex(self, s: str) -> str:
        # collapse any whitespace sequence to \s+
        collapsed = re.sub(r"\s+", "\\s+", s.strip())
        return collapsed

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
        new_lines = lines[:line] + text.splitlines() + lines[line:]
        new_content = "\n".join(new_lines) + ("\n" if current.endswith("\n") else "")
        self._write_file(path, new_content)
        snippet = "\n".join(
            self._numbered(
                new_lines[
                    max(0, line - SNIPPET_LINES) : line
                    + SNIPPET_LINES
                    + len(text.splitlines())
                ],
                offset=max(1, line - SNIPPET_LINES + 1),
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
        return [f"{i + offset:6}\t{l}" for i, l in enumerate(lines)]

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


# ---------------------------------------------------------------------------
# Module‑level helpers
# ---------------------------------------------------------------------------


def _which(exe: str) -> Optional[str]:
    """Return path to executable or None."""
    return shutil.which(exe)


def _truncate(text: str, max_len: int = MAX_STDOUT) -> str:
    """Truncate long strings and indicate omission."""
    return text if len(text) <= max_len else text[:max_len] + "\n⋯ (truncated)"
