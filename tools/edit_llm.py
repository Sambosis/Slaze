"""
edit_llm.py - LLM-enhanced file editor
========================================
File editor that uses LLM calls for intelligent text replacement and insertion.

Commands
--------
view • create • str_replace • insert • undo_edit • lint • format

Key Features
~~~~~~~~~~~~
* **LLM-powered replacement** (`str_replace`) - Uses LLM to understand context
  and make intelligent replacements based on natural language instructions.
  
* **LLM-powered insertion** (`insert`) - Uses LLM to generate and insert
  appropriate code/text based on context and instructions.

* Atomic, lock-safe writes with `portalocker`, size-guarded reads
  (≤ 512 KiB per file).

* Built-in lint (`ruff`→`pylint`) and formatter (`black`).

* Strict path sandboxing: files stay under `REPO_DIR`.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import portalocker  # type: ignore

from .base import BaseTool, ToolError, ToolResult
from config import get_constant, MAIN_MODEL
from utils.llm_client import create_llm_client

# ---------------------------------------------------------------------------
# Configuration & constants
# ---------------------------------------------------------------------------

_LOG = logging.getLogger(__name__)
MAX_FILE_BYTES = 512 * 1024  # 512 KiB file size cap
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


class EditLLMTool(BaseTool):
    """Cross-platform editor with LLM-enhanced editing capabilities."""

    api_type = "text_editor_llm_20250124"

    @property
    def name(self) -> str:  # satisfies BaseTool contract
        return "str_replace_editor_llm"

    @property
    def description(self) -> str:  # satisfies BaseTool contract
        return (
            "An LLM-enhanced file editor that can view, create, edit, lint, and "
            "format files. Uses AI to understand context and make intelligent "
            "replacements and insertions based on natural language instructions."
        )

    # ------------------------------------------------------------------
    # Construction / schema
    # ------------------------------------------------------------------

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display)
        self.display = display
        self._repo_dir: Path = Path(get_constant("REPO_DIR")).resolve()
        self._file_history: Dict[Path, List[str]] = defaultdict(list)
        self._llm_client = create_llm_client(MAIN_MODEL)

    def to_params(self) -> Dict[str, Any]:
        """Expose the OpenAI function-calling schema."""
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
                        "old_str": {"type": "string", "description": "For str_replace: description of what to replace or pattern to find"},
                        "new_str": {"type": "string", "description": "For str_replace/insert: description of changes or new content to add"},
                        "insert_line": {"type": "integer"},
                        "match_mode": {
                            "type": "string",
                            "enum": ["exact", "regex", "fuzzy", "llm"],
                            "description": "Replacement strategy for str_replace (default llm for intelligent replacement).",
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
        match_mode: str = "llm",
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
        }
        try:
            if cmd_enum is Command.CREATE:
                result = self._cmd_create(_path, file_text)
            elif cmd_enum is Command.VIEW:
                result = await self._cmd_view(_path, view_range)
            elif cmd_enum is Command.STR_REPLACE:
                result = await self._cmd_str_replace_llm(_path, old_str, new_str, match_mode)
            elif cmd_enum is Command.INSERT:
                result = await self._cmd_insert_llm(_path, insert_line, new_str)
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
            _LOG.error("EditLLMTool failure", exc_info=True)
            error_result = ToolResult(
                output=f"EditLLMTool error running {command} on {_path}: {exc}",
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
    # LLM-enhanced command handlers
    # ------------------------------------------------------------------

    async def _cmd_str_replace_llm(
        self, path: Path, old_desc: Optional[str], new_desc: Optional[str], mode: str
    ) -> ToolResult:
        """LLM-enhanced string replacement."""
        if mode != "llm":
            # Fall back to traditional replacement for non-LLM modes
            return await self._cmd_str_replace_traditional(path, old_desc, new_desc, mode)

        if not old_desc and not new_desc:
            raise ToolError("At least one of old_str or new_str required for str_replace")

        # Prefer deterministic replacement when both pieces look like concrete code. This keeps
        # simple replacements reliable while still allowing natural-language edits to use the LLM.
        if old_desc and self._looks_like_code(old_desc) and (
            new_desc is None or self._looks_like_code(new_desc)
        ):
            try:
                return self._file_str_replace(path, old_desc, new_desc or "", "exact")
            except ToolError as err:
                message = str(err)
                # If we fail to find a match, retry with fuzzy matching before escalating to LLM
                if "No match" in message or "No match found" in message:
                    try:
                        return self._file_str_replace(path, old_desc, new_desc or "", "fuzzy")
                    except ToolError:
                        # Fall through to LLM behaviour
                        pass
                elif "Multiple matches" in message:
                    # Let the LLM handle ambiguous matches where possible
                    pass
                else:
                    # Bubble up any other errors
                    raise

        # Read the current file content
        current_content = self._read_file(path)

        # Prepare the prompt for the LLM
        prompt = self._create_replacement_prompt(current_content, old_desc, new_desc)

        # Call the LLM to get the modified content
        try:
            modified_content = await self._call_llm_for_edit(prompt)

            # Save the history before writing
            self._file_history[path].append(current_content)

            # Write the modified content
            self._write_file(path, modified_content)

            # Create a summary of changes
            summary = self._create_change_summary(
                current_content, modified_content, old_desc, new_desc
            )

            return ToolResult(output=f"Modified {path} using LLM\n{summary}")

        except Exception as e:
            raise ToolError(f"LLM replacement failed: {str(e)}")

    async def _cmd_insert_llm(
        self, path: Path, line: Optional[int], instruction: Optional[str]
    ) -> ToolResult:
        """LLM-enhanced text insertion."""
        if line is None or instruction is None:
            raise ToolError("insert_line and new_str (instruction) required for insert")
        
        # Read the current file content
        current_content = self._read_file(path)
        lines = current_content.splitlines()
        
        if line < 0:
            raise ToolError("insert_line must be non-negative")
        if line > len(lines):
            raise ToolError("insert_line beyond EOF")
        
        # Get context around the insertion point
        context_before = "\n".join(lines[max(0, line-10):line])
        context_after = "\n".join(lines[line:min(len(lines), line+10)])
        
        # Create prompt for the LLM
        prompt = self._create_insertion_prompt(
            current_content, line, instruction, context_before, context_after
        )
        
        try:
            # Call the LLM to get the complete modified file
            modified_content = await self._call_llm_for_edit(prompt)
            
            # Save the history before writing
            self._file_history[path].append(current_content)
            
            # Write the modified content
            self._write_file(path, modified_content)
            
            # Create a summary
            new_lines = modified_content.splitlines()
            snippet_start = max(0, line - SNIPPET_LINES)
            snippet_end = min(len(new_lines), line + SNIPPET_LINES + 5)
            snippet = "\n".join(
                self._numbered(
                    new_lines[snippet_start:snippet_end],
                    offset=snippet_start + 1,
                )
            )
            
            return ToolResult(output=f"Inserted text in {path} at line {line}\n{snippet}")
            
        except Exception as e:
            raise ToolError(f"LLM insertion failed: {str(e)}")

    async def _cmd_str_replace_traditional(
        self, path: Path, old: Optional[str], new: Optional[str], mode: str
    ) -> ToolResult:
        """Traditional string replacement (exact, regex, fuzzy)."""
        if old is None:
            raise ToolError("`old_str` required for str_replace")
        if mode not in {"exact", "regex", "fuzzy"}:
            mode = "exact"

        try:
            return self._file_str_replace(path, old, new or "", mode)
        except ToolError as err:
            # If exact/regex match fails, try fuzzy matching
            message = str(err)
            if mode in {"exact", "regex"} and ("No match" in message or "No match found" in message):
                fallback_result = self._file_str_replace(path, old, new or "", "fuzzy")
                if fallback_result.output:
                    fallback_result.output = (
                        "[fallback to fuzzy match]\n" + fallback_result.output
                    )
                return fallback_result
            raise

    # ------------------------------------------------------------------
    # LLM helper methods
    # ------------------------------------------------------------------

    def _create_replacement_prompt(self, content: str, old_desc: str, new_desc: str) -> str:
        """Create a prompt for LLM-based replacement."""
        prompt = f"""You are a code editor assistant. Your task is to modify the given file according to the instructions.

IMPORTANT: You must output ONLY the complete modified file content, with no explanations, no markdown code blocks, and no additional text.

Current file content:
{content}

Instructions:
"""
        if old_desc:
            prompt += f"Find and replace: {old_desc}\n"
        if new_desc:
            prompt += f"Replace with/Change to: {new_desc}\n"
        
        prompt += "\nOutput the complete modified file content:"
        return prompt

    def _create_insertion_prompt(
        self, content: str, line: int, instruction: str, context_before: str, context_after: str
    ) -> str:
        """Create a prompt for LLM-based insertion."""
        prompt = f"""You are a code editor assistant. Your task is to insert new content into a file at a specific location.

IMPORTANT: You must output ONLY the complete modified file content, with no explanations, no markdown code blocks, and no additional text.

Current file content:
{content}

Insertion point: Line {line}

Context before insertion point:
{context_before}

Context after insertion point:
{context_after}

Instruction for what to insert:
{instruction}

Output the complete modified file content with the new content inserted at line {line}:"""
        return prompt

    async def _call_llm_for_edit(self, prompt: str) -> str:
        """Call the LLM and get the edited content."""
        messages = [
            {
                "role": "system",
                "content": "You are a precise code editor. Output only the complete modified file content with no additional text or formatting."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Use higher token limit for file editing
        response = await self._llm_client.call(
            messages=messages,
            max_tokens=8000,  # Allow for larger files
            temperature=0.1    # Low temperature for precise editing
        )
        
        # Clean up the response - remove any markdown code blocks if present
        cleaned = response.strip()
        
        # Remove markdown code blocks if they were added despite instructions
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Find the first line that's just ```
            start_idx = 0
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                if not line.startswith("```"):
                    start_idx = i
                    break
            # Find the last line that's just ```
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            cleaned = "\n".join(lines[start_idx:end_idx])
        
        return cleaned

    def _create_change_summary(
        self,
        original: str,
        modified: str,
        old_desc: Optional[str],
        new_desc: Optional[str],
    ) -> str:
        """Create a summary of changes made."""
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()
        
        # Simple diff summary
        lines_added = len(modified_lines) - len(original_lines)
        
        summary = []
        if old_desc:
            summary.append(f"Searched for: {old_desc}")
        if new_desc:
            summary.append(f"Replaced with: {new_desc}")
        
        if lines_added > 0:
            summary.append(f"Lines added: {lines_added}")
        elif lines_added < 0:
            summary.append(f"Lines removed: {-lines_added}")
        else:
            summary.append("Lines modified in place")
        
        return "\n".join(summary)

    def _looks_like_code(self, text: str) -> bool:
        """Heuristically determine whether the provided text resembles source code."""
        if text is None:
            return False

        snippet = text.strip()
        if not snippet:
            return False

        # Multi-line strings are almost always code snippets.
        if "\n" in snippet:
            return True

        lowered = snippet.lower()
        code_keywords = [
            "def ",
            "class ",
            "return",
            "import ",
            "from ",
            "async ",
            "await ",
            "if ",
            "elif ",
            "else:",
            "for ",
            "while ",
            "try:",
            "except",
            "with ",
            "switch",
            "case ",
            "function ",
            "var ",
            "let ",
            "const ",
        ]
        if any(keyword in lowered for keyword in code_keywords):
            return True

        # Count the presence of non-alphabetic characters that frequently appear in code.
        code_punctuation = set("{}[]();=:+-*/.<>,""'\\")
        punctuation_score = sum(1 for ch in snippet if ch in code_punctuation)
        if punctuation_score >= 2:
            return True

        # Detect common identifier patterns such as snake_case or camelCase.
        has_underscore = "_" in snippet
        has_camel_case = any(
            ch.islower() and idx + 1 < len(snippet) and snippet[idx + 1].isupper()
            for idx, ch in enumerate(snippet)
        )
        if has_underscore or has_camel_case:
            return True

        # Basic assignment or attribute access patterns (e.g., "value = 1").
        if re.match(r"[A-Za-z_][\w\.\[\]\s]*=", snippet):
            return True

        # Short single-word snippets are likely identifiers rather than prose.
        if len(snippet.split()) == 1 and len(snippet) <= 40:
            return True

        return False

    # ------------------------------------------------------------------
    # Standard command handlers (unchanged from original)
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

    def _cmd_undo(self, path: Path) -> ToolResult:
        if not self._file_history[path]:
            raise ToolError("No edits to undo for this file")
        prev = self._file_history[path].pop()
        self._write_file(path, prev)
        return ToolResult(output=f"Undo successful for {path}")

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
    # Traditional file replacement methods (for fallback)
    # ------------------------------------------------------------------

    def _file_str_replace(
        self, path: Path, old: str, new: str, mode: str
    ) -> ToolResult:
        text = self._read_file(path)
        pattern: str
        flags = re.DOTALL
        if mode == "exact":
            pattern = re.escape(old)
            pattern = self._crlf_tolerant(pattern)
        elif mode == "regex":
            pattern = self._crlf_tolerant(old)
        else:  # fuzzy
            pattern = self._build_fuzzy_regex(old)
        matches = list(re.finditer(pattern, text, flags))
        if not matches:
            raise ToolError("No match found for replacement")
        if len(matches) > 1:
            line_numbers: List[int] = []
            for m in matches:
                line_numbers.append(text.count("\n", 0, m.start()) + 1)
            raise ToolError(
                "Multiple matches found; make pattern more specific. Matches at lines: "
                + ", ".join(str(n) for n in line_numbers)
            )
        m = matches[0]
        candidate_text = text[: m.start()] + new + text[m.end() :]
        normalized_text = self._normalize_to_original_newlines(original=text, modified=candidate_text)
        self._write_file(path, normalized_text)
        snippet = self._snippet(normalized_text, m.start(), m.start() + len(new))
        return ToolResult(output=f"Replaced code in {path}\n{snippet}")

    def _build_fuzzy_regex(self, s: str) -> str:
        """Construct a regex that matches the given string with flexible whitespace."""
        parts = re.findall(r"\s+|\S+", s)
        escaped_segments: List[str] = []
        for part in parts:
            if part.isspace():
                escaped_segments.append(r"\s+")
            else:
                escaped_segments.append(re.escape(part))
        pattern = "".join(escaped_segments)
        pattern = re.sub(r"(?:\\s\+){2,}", r"\\s+", pattern)
        return pattern

    def _crlf_tolerant(self, pattern: str) -> str:
        """Convert literal newlines in a pattern to match either LF or CRLF."""
        pattern = pattern.replace("\r\n", r"\r?\n")
        pattern = pattern.replace("\n", r"\r?\n")
        return pattern

    # ------------------------------------------------------------------
    # Display helpers
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
                self.display.add_message("assistant", formatted)
        except Exception:
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
        """Return a fenced console block showing the invoked edit command and its output."""
        try:
            rel_path: str
            try:
                rel_path = str(path.relative_to(self._repo_dir))
            except Exception:
                rel_path = str(path)
            mode_arg = ""
            if call_args and cmd_enum is Command.STR_REPLACE:
                mm = call_args.get("match_mode") or "llm"
                mode_arg = f" --mode={mm}"
            invoked = f"$ edit {cmd_enum.value} {rel_path}{mode_arg}".rstrip()
            lines: List[str] = ["```console", invoked]

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

                lines.extend(_preview("instruction", call_args.get("old_str")))
                if call_args.get("new_str"):
                    lines.extend(_preview("replacement", call_args.get("new_str")))
                    
            out_text = result.output or ""
            err_text = result.error or ""

            combined = out_text
            if err_text and err_text not in combined:
                if combined:
                    combined = combined.rstrip() + ("\n" if not combined.endswith("\n") else "") + err_text
                else:
                    combined = err_text

            if is_error and not err_text and not out_text:
                combined = "Error occurred but no output captured"

            truncated = False
            if len(combined) > max_chars:
                combined = (
                    combined[: max_chars // 2]
                    + "\n… (truncated) …\n"
                    + combined[-max_chars // 2 :]
                )
                truncated = True
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
    # Path and file helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, p: str | Path) -> Path:
        p = Path(p)
        if not p.is_absolute():
            p = self._repo_dir / p
        p = p.resolve()
        if not str(p).startswith(str(self._repo_dir)):
            raise ToolError("Path escapes REPO_DIR sandbox")
        return p

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
        with portalocker.Lock(str(tmp), "wb", timeout=5) as fp:
            fp.write(content.encode("utf-8"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak.{timestamp}")
        if path.exists():
            shutil.copy2(path, backup)
            self._file_history[path].append(self._read_file(path))
        shutil.move(tmp, path)
        from utils.file_logger import log_file_operation
        log_file_operation(path, "modify")

    def _dir_list(self, root: Path) -> str:
        entries: List[str] = []
        for depth in range(3):
            glob = os.path.join(*["*"] * (depth + 1)) if depth else "*"
            for item in root.glob(glob):
                if any(part.startswith(".") for part in item.parts):
                    continue
                entries.append(str(item.relative_to(self._repo_dir)))
        return "\n".join(sorted(entries))

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

    def _numbered(self, lines: List[str], *, offset: int = 1) -> List[str]:
        return [f"{idx + offset:6}\t{line}" for idx, line in enumerate(lines)]

    def _snippet(self, text: str, start: int, end: int) -> str:
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

    def _detect_original_newline(self, text: str) -> Tuple[str, bool]:
        """Detect the predominant newline sequence and trailing newline policy."""
        uses_crlf = "\r\n" in text
        stray_lf_present = "\n" in text.replace("\r\n", "")
        newline_seq = "\r\n" if uses_crlf and not stray_lf_present else "\n"
        trailing = text.endswith(("\r\n", "\n"))
        return newline_seq, trailing

    def _normalize_to_original_newlines(self, *, original: str, modified: str) -> str:
        """Normalize newline characters in modified text to match the original file."""
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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _which(exe: str) -> Optional[str]:
    """Return path to executable or None."""
    return shutil.which(exe)


def _truncate(text: str, max_len: int = MAX_STDOUT) -> str:
    """Truncate long strings and indicate omission."""
    return text if len(text) <= max_len else text[:max_len] + "\n⋯ (truncated)"