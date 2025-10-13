# write_code Tool README

## Overview
The `write_code` tool is responsible for materializing files on disk based on structured
instructions from the agent orchestration layer. It is optimized for generating small batches of
files (up to five at a time), supports both source code and binary assets, and coordinates with the
OpenRouter API to synthesize implementation details. When attached to an interactive UI, it also
streams formatted console updates so a human operator can follow progress in real time.

## When to Use This Tool
Use `WriteCodeTool` whenever the agent must create or overwrite files as part of a coding task. The
tool accepts rich descriptions for each file, optional import guidance, and can route requests for
image assets to the dedicated `PictureGenerationTool`. Because it orchestrates retries, logging, and
LLM selection internally, most higher-level agent logic only needs to supply accurate file
specifications.

## Invocation Interface
`WriteCodeTool` implements the function-calling contract required by the orchestrator. It exposes a
single command, `write_codebase`, with the following parameters:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `command` | enum | ✅ | Must be `"write_codebase"`. Any other value is rejected before work begins. |
| `files` | array<FileDetail> | ✅ | Up to five file descriptors. Each descriptor includes a `filename`,
  a `code_description`, and optional `external_imports`/`internal_imports` arrays. |

The tool returns a `ToolResult` containing a JSON payload that summarizes status, counters for code
and image files, and detailed per-file outcomes. Errors are surfaced both in the return value and in
the streamed console messages when a display is attached. 【F:tools/write_code.py†L201-L289】【F:tools/write_code.py†L638-L705】

### FileDetail Validation
Incoming payloads are validated with a strict `pydantic` model. The validation step enforces the
presence of the filename and description fields, rejects unexpected keys, and prevents the execution
path from starting if no files are supplied. 【F:tools/write_code.py†L164-L208】

## High-Level Execution Flow
1. **Resolve host write path** – The tool obtains `REPO_DIR` from the configuration layer and ensures
   the directory exists before any I/O. This is the only write target; all file paths are resolved
   relative to it. 【F:tools/write_code.py†L234-L248】
2. **Partition targets** – File descriptors are split into image and code buckets by examining the
   file extension. Image requests are delegated to `PictureGenerationTool`; only the remaining entries
   proceed through the code pipeline. 【F:tools/write_code.py†L273-L347】
3. **Generate code asynchronously** – Descriptions for code files are processed concurrently. Each
   task synthesizes implementation text using one or more language models (described below). 【F:tools/write_code.py†L356-L404】
4. **Write artifacts** – Generated strings are normalized with `ftfy`, written to the host filesystem,
   and recorded in the file operation log. The UI receives rich terminal-style summaries and syntax
   highlighted previews. 【F:tools/write_code.py†L431-L569】【F:tools/write_code.py†L1301-L1327】
5. **Summarize results** – A final status message combines successes and failures across code and
   image work, ensuring the caller receives aggregate counts as well as per-file diagnostics.
   【F:tools/write_code.py†L607-L706】

## LLM Orchestration Strategy
`WriteCodeTool` is intentionally redundant: it fans out requests across multiple OpenRouter models and
falls back to a robust single-model pipeline when necessary.

* **Parallel generation** – For each file, `CODE_LIST` is iterated and every model generates a candidate
  implementation concurrently. Failures are logged but do not abort the batch. 【F:tools/write_code.py†L894-L944】
* **Quality selection** – If more than one candidate succeeds, a final arbiter call using `CODE_MODEL`
  evaluates the variants and returns the highest-quality version. Invalid responses default to the
  first viable candidate. 【F:tools/write_code.py†L945-L1016】
* **Retry-hardened fallback** – When all speculative calls fail, the tool reverts to the original
  `_call_llm_to_generate_code_original` pipeline. That path employs exponential backoff via Tenacity,
  retries transient OpenAI or network errors, and aggressively validates the returned code block.
  【F:tools/write_code.py†L1018-L1188】
* **Context enrichment** – Prompts include the overall task description, recent file-operation logs,
  and import hints so that the generated code fits the surrounding project. 【F:tools/write_code.py†L906-L924】【F:tools/write_code.py†L1048-L1080】

Environment configuration is critical: every LLM call requires an `OPENROUTER_API_KEY`, and the tool
resolves the active model set from `config.py` constants (`CODE_MODEL`, `CODE_LIST`, optional
`CODE_GEN_MODEL`). 【F:config.py†L58-L110】【F:tools/write_code.py†L965-L976】【F:tools/write_code.py†L1119-L1178】

## File Writing and Logging Behavior
* **Text normalization** – Raw completions run through `ftfy.fix_text` to repair encoding artifacts
  before being persisted. 【F:tools/write_code.py†L506-L514】
* **Path management** – Absolute host paths are computed, verified, and then converted back to Docker
  paths for UI display if necessary. 【F:tools/write_code.py†L448-L486】
* **Audit trail** – Every write is logged via `log_file_operation`, and generated content is appended to
  a dedicated code log when configured. 【F:tools/write_code.py†L487-L559】【F:tools/write_code.py†L1244-L1267】
* **Rich feedback** – When a display is provided, the tool streams terminal-style updates and syntax
  highlighted HTML previews to help operators inspect generated files quickly.
  【F:tools/write_code.py†L512-L538】【F:tools/write_code.py†L1270-L1299】

## Image Generation Path
Image filenames (e.g., `.png`, `.svg`) are intercepted early and processed through the
`PictureGenerationTool`. Image prompts reuse the `code_description` text, and the results are merged
back into the final `ToolResult` alongside code summaries. 【F:tools/write_code.py†L259-L347】【F:tools/write_code.py†L666-L705】

## Output Contract
On success, the tool emits a JSON-formatted payload describing:

* Overall status (`success`, `partial_success`, or `error`)
* Total files processed and succeeded
* Separate counts for code and image artifacts
* The resolved write path
* A list of per-file results, each noting success, operation type, and any captured error messages

This payload is serialized through `format_output`, the helper provided by the base tool class, and is
always accompanied by UI notifications when a display is connected. 【F:tools/write_code.py†L607-L706】

## Potential Bugs and Edge Cases
* **Missing OpenRouter key** – The tool raises a `ValueError` if `OPENROUTER_API_KEY` is absent. When
  invoked inside `__call__`, this propagates to the generic exception handler and surfaces as a
  critical failure message. Consider supplying a clearer pre-flight check. 【F:tools/write_code.py†L1119-L1178】
* **Model selection loops** – If every model in `CODE_LIST` fails with an exception, the fallback path
  retries the same models sequentially. Misconfigured credentials could therefore waste time before
  ultimately returning a failure. 【F:tools/write_code.py†L941-L1188】
* **Concurrent file writes** – Multiple asynchronous tasks write to disk without locking. In the rare
  case where two descriptors target the same filename, the last one wins silently. 【F:tools/write_code.py†L431-L569】
* **UI coupling** – `self.display` is assumed to accept rich HTML snippets. In headless environments,
  these formatting calls are skipped, but any unexpected display implementation could still raise.
  【F:tools/write_code.py†L512-L538】【F:tools/write_code.py†L1270-L1299】

## Suggested Improvements
1. **Pre-flight diagnostics** – Add a lightweight readiness check (environment variables, directory
   permissions, model availability) so the tool can fail fast before kicking off asynchronous work.
2. **Dynamic batch sizing** – The tool currently trusts the caller to enforce the “five files” limit
   mentioned in the description. Explicit enforcement with a descriptive error would make the contract
   clearer. 【F:tools/write_code.py†L213-L226】
3. **Configurable image options** – Allow callers to control image dimensions or format rather than
   hard-coding 1024×1024 outputs. 【F:tools/write_code.py†L307-L328】
4. **Enhanced diff safety** – Integrate optional read-modify-write safeguards (e.g., verifying the file
   has not changed since generation) to better support collaborative editing scenarios. 【F:tools/write_code.py†L431-L569】
5. **Telemetry hooks** – Expose metrics about model success rates and retry counts to help tune the
   `CODE_LIST` ordering and improve latency. 【F:tools/write_code.py†L894-L944】【F:tools/write_code.py†L1119-L1178】

