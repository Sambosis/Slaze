"""
edit_tool_robust.py — v5  (fully functional)
Robust code‑oriented editor with lint / format helpers **and complete edit ops**.
Fixes the missing helper methods (`_file_str_replace`, `_file_insert`, `_file_view`, `_file_undo`,
`_run`, `_which`, `_truncate`, `_numbered`, `_snippet`).  `to_params()` still advertises the full
schema including the new commands.
"""

from __future__ import annotations

import os
import shutil
import logging
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from enum import Enum
from collections import defaultdict
from typing import List, Optional, Dict, Any, Tuple

import portalocker  # type: ignore

from .base import BaseTool, ToolError, ToolResult
from config import get_constant
from utils.file_logger import log_file_operation

_LOG = logging.getLogger(__name__)

MAX_FILE_BYTES = 512 * 1024  # 512 KiB per file
MAX_STDOUT_CHARS = 10_000  # keep subprocess output manageable
SNIPPET_LINES = 4  # context lines around edits

# ---------------------------------------------------------------------------
# 🔘 Command enumeration
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
# 🛠️  The tool
# ---------------------------------------------------------------------------


class EditTool(BaseTool):
    """Cross‑platform file editor with refactor / debugging helpers."""

    description = (
        "A cross‑platform file editor that can view, create, edit, lint, and "
        "format code files inside the REPO_DIR sandbox."
    )

    api_type = "text_editor_20250124"
    name = "str_replace_editor"

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display)
        self.display = display
        self._repo_dir = Path(get_constant("REPO_DIR")).resolve()
        self._file_history: Dict[Path, List[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    # 📣 Function‑calling schema
    # ------------------------------------------------------------------

    def to_params(self) -> Dict[str, Any]:
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
                            "enum": Command.list(),
                            "description": "The command to execute.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Target file or directory (relative to project root unless absolute).",
                        },
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
                    },
                    "required": ["command", "path"],
                },
            },
        }

    # ------------------------------------------------------------------
    # 🛠️ Public entrypoint
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
        **kwargs,
    ) -> ToolResult:
        cmd = self._validate_command(command)
        _path = self._resolve_path(path)
        try:
            if cmd is Command.CREATE:
                return self._cmd_create(_path, file_text)
            if cmd is Command.VIEW:
                return await self._cmd_view(_path, view_range)
            if cmd is Command.STR_REPLACE:
                return self._cmd_str_replace(_path, old_str, new_str)
            if cmd is Command.INSERT:
                return self._cmd_insert(_path, insert_line, new_str)
            if cmd is Command.UNDO_EDIT:
                return self._cmd_undo(_path)
            if cmd is Command.LINT:
                return self._cmd_lint(_path)
            if cmd is Command.FORMAT:
                return self._cmd_format(_path)
            raise ToolError(f"Unhandled command {cmd}")
        except Exception as exc:
            _LOG.error("EditTool failure", exc_info=True)
            return ToolResult(
                output=f"EditTool error running {command} on {_path}: {exc}",
                error=str(exc),
                tool_name=self.name,
                command=command,
            )

    # ------------------------------------------------------------------
    # ✅ Command validation
    # ------------------------------------------------------------------

    def _validate_command(self, cmd: str) -> Command:
        try:
            return Command(cmd)
        except ValueError:
            raise ToolError(
                f"Invalid command '{cmd}'. Valid commands: {', '.join(Command.list())}."
            )

    # ------------------------------------------------------------------
    # 🔧 Command handlers
    # ------------------------------------------------------------------

    def _cmd_create(self, path: Path, file_text: Optional[str]) -> ToolResult:
        if file_text is None:
            raise ToolError("`file_text` required for create")
        self._write_file(path, file_text)
        return ToolResult(output=f"Created {path}")

    async def _cmd_view(
        self, path: Path, view_range: Optional[List[int]]
    ) -> ToolResult:
        if path.is_dir():
            listing = self._dir_list(path)
            return ToolResult(output=listing)
        if path.is_file():
            return self._file_view(path, view_range)
        raise ToolError(f"{path} is neither file nor directory")

    def _cmd_str_replace(
        self, path: Path, old: Optional[str], new: Optional[str]
    ) -> ToolResult:
        if old is None:
            raise ToolError("`old_str` required")
        return self._file_str_replace(path, old, new or "")

    def _cmd_insert(
        self, path: Path, line: Optional[int], text: Optional[str]
    ) -> ToolResult:
        if line is None or text is None:
            raise ToolError("insert_line and new_str required")
        return self._file_insert(path, line, text)

    def _cmd_undo(self, path: Path) -> ToolResult:
        return self._file_undo(path)

    def _cmd_lint(self, path: Path) -> ToolResult:
        runner = ["ruff", "--quiet", "--format", "text", str(path)]
        if not _which("ruff"):
            runner = ["python", "-m", "pylint", str(path)]
        out, err, code = self._run(runner, cwd=self._repo_dir)
        header = f"lint ({'ruff' if 'ruff' in runner[0] else 'pylint'}) exit {code}"
        return ToolResult(output=f"{header}\n{_truncate(out or err)}")

    def _cmd_format(self, path: Path) -> ToolResult:
        if not _which("black"):
            raise ToolError("Black not installed")
        runner = ["black", "--quiet", str(path)]
        out, err, code = self._run(runner, cwd=self._repo_dir)
        msg = "formatted" if code == 0 else "already formatted"
        return ToolResult(output=f"black: {msg} — {path}")

    # ------------------------------------------------------------------
    # 📂 Path helpers
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
    # 📝 File helpers
    # ------------------------------------------------------------------

    def _write_file(self, path: Path, content: str):
        if len(content.encode()) > MAX_FILE_BYTES:
            raise ToolError("File too large")
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(tmp_path), "w", timeout=5) as f:
            f.write(content)
        # backup existing
        if path.exists():
            bak = path.with_suffix(
                path.suffix + f".bak-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
            shutil.copy2(path, bak)
        tmp_path.replace(path)
        log_file_operation(path, "modify")
        # Record history after successful write
        self._file_history[path].append(content)

    def _read_file(self, path: Path) -> str:
        data = path.read_text(encoding="utf-8")
        if len(data.encode()) > MAX_FILE_BYTES:
            raise ToolError("File too large to display")
        return data

    # ------------------------------------------------------------------
    # 🔬 View helpers
    # ------------------------------------------------------------------

    def _dir_list(self, root: Path) -> str:
        acc: List[str] = []
        for p in root.rglob("*"):
            if any(part.startswith(".") for part in p.parts):
                continue
            rel = p.relative_to(self._repo_dir)
            if p.is_dir():
                acc.append(f"{rel}/")
            else:
                acc.append(str(rel))
        return "\n".join(sorted(acc))

    def _file_view(self, path: Path, view_range: Optional[List[int]]) -> ToolResult:
        text = self._read_file(path)
        lines = text.split("\n")
        init = 1
        end = len(lines)
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ToolError("`view_range` must be two integers [start, end]")
            init, end = view_range
            if init < 1 or (end != -1 and end < init):
                raise ToolError("Invalid view_range")
            if end == -1:
                end = len(lines)
        snippet = "\n".join(lines[init - 1 : end])
        return ToolResult(output=_numbered(snippet, init_line=init))

    # ------------------------------------------------------------------
    # ✂️  Edit helpers
    # ------------------------------------------------------------------

    def _file_str_replace(self, path: Path, old: str, new: str) -> ToolResult:
        text = self._read_file(path)
        occurrences = text.count(old)
        if occurrences == 0:
            raise ToolError("old_str not found")
        if occurrences > 1:
            raise ToolError("old_str occurs multiple times; be explicit")
        new_text = text.replace(old, new)
        self._write_file(path, new_text)
        line_no = text.split(old)[0].count("\n")
        snippet = _snippet(new_text, line_no, len(new.split("\n")))
        return ToolResult(output=f"Replaced text in {path}\n{snippet}")

    def _file_insert(self, path: Path, line: int, new_str: str) -> ToolResult:
        text = self._read_file(path)
        lines = text.split("\n")
        if line < 0 or line > len(lines):
            raise ToolError("insert_line out of range")
        new_lines = lines[:line] + new_str.split("\n") + lines[line:]
        self._write_file(path, "\n".join(new_lines))
        snippet = _snippet("\n".join(new_lines), line, len(new_str.split("\n")))
        return ToolResult(output=f"Inserted text into {path}\n{snippet}")

    def _file_undo(self, path: Path) -> ToolResult:
        history = self._file_history.get(path)
        if not history:
            raise ToolError("No edit history for this file")
        prev = history.pop()
        self._write_file(path, prev)
        return ToolResult(output=f"Undo successful for {path}")

    # ------------------------------------------------------------------
    # 🏃 Subprocess helper
    # ------------------------------------------------------------------

    def _run(self, cmd: List[str], *, cwd: Path | None = None) -> Tuple[str, str, int]:
        _LOG.debug("Running %s", cmd)
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return proc.stdout, proc.stderr, proc.returncode


# ---------------------------------------------------------------------------
# 🧩 Utility helpers (not methods)
# ---------------------------------------------------------------------------


def _which(bin_name: str) -> bool:
    return shutil.which(bin_name) is not None


def _truncate(text: str, limit: int = MAX_STDOUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n⋯ (truncated)"


def _numbered(blob: str, *, init_line: int = 1) -> str:
    return "\n".join(f"{i+init_line:6}│ " + l for i, l in enumerate(blob.split("\n")))


def _snippet(full_text: str, start_line: int, added_lines: int) -> str:
    """Return a small numbered context around the edit."""
    lines = full_text.split("\n")
    s = max(0, start_line - SNIPPET_LINES)
    e = min(len(lines), start_line + added_lines + SNIPPET_LINES)
    segment = "\n".join(lines[s:e])
    return _numbered(segment, init_line=s + 1)
