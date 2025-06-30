# Slaze

Slaze is an experimental agent that can generate code, set up projects and execute tools using large language models. The entry point for running the agent from the console is **`run.py`**.

## Features

- Uses OpenAI or OpenRouter API for LLM interaction.
- Supports a collection of tools such as Bash commands, project setup, code generation and more.
- Prompts are stored in the `prompts/` directory and can be selected or created when the app starts.
- Logs are saved under `logs/`.
- Interactive prompt creation: you can edit existing prompt files or create a new one on launch.
- New prompts may be sent to the LLM for analysis to generate an expanded task definition when an
  `OPENROUTER_API_KEY` is available.
- Tools include `WriteCodeTool`, `ProjectSetupTool`, `BashTool`, `DockerEditTool` and
  `PictureGenerationTool` for generating files and images inside a Docker container.
- Tool execution results are logged to `logs/tool.log` and `logs/tool.txt` for later review.

## Installation

1. Install Python 3.12 (see `.python-version`).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Provide an API key via environment variables or a `.env` file in the project root:

```
OPENROUTER_API_KEY=...  # or OPENAI_API_KEY
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # optional override for OpenRouter
```

## Running the console app

Execute:

```bash
python run.py console
```

## Running the web UI

To use the interactive web interface, run:

```bash
python run.py web
```

By default the server starts on port 5000. Open your browser to:

```
http://localhost:5000/
```

For a detailed walkthrough see [docs/web_interface_guide.md](docs/web_interface_guide.md).

The script loads environment variables with `dotenv` and launches the `AgentDisplayConsole`. You will be prompted to choose a prompt from `prompts/` or create a new one. After choosing a task, the agent runs, using tools and displaying output directly in the terminal.

When creating a new prompt you can edit the text before saving. If the
`OPENROUTER_API_KEY` environment variable is set, the prompt text is first sent
to an LLM to produce a more detailed task description. A project directory under
`repo/<prompt>` is then created and the final task is saved to `logs/task.txt`.
Tool activity and agent messages are streamed to the console and also logged to
`logs/` for later reference.

## Logging

This project uses Python's built-in `logging` module for application logging.

**Configuration**:
*   Logging is configured centrally in `config.py`.
*   Key settings include console logging (level controlled by the `LOG_LEVEL_CONSOLE` constant, typically "INFO") and rotating file logging to `logs/app.log` (level controlled by the `LOG_LEVEL_FILE` constant, typically "DEBUG"). Log rotation (based on size and backup count) is set up for `logs/app.log`.

**Usage**:
*   To use logging in a module, obtain a logger instance:
    ```python
    import logging
    logger = logging.getLogger(__name__)
    ```
*   Example log messages:
    ```python
    logger.debug("Detailed debug information.")
    logger.info("An informational message.")
    logger.warning("A warning occurred.")
    logger.error("An error occurred.")
    logger.critical("A critical error occurred.")
    ```
*   To log exceptions:
    ```python
    try:
        # ... some operation ...
        pass
    except Exception as e:
        logger.error(f"Operation failed: {e}", exc_info=True)
    # Or, if the message is sufficient and you want the stack trace:
    # logger.exception("Operation failed")
    ```

**Specialized File Operation Logging**:
*   For auditing file creations, modifications, and deletions, the project uses a specialized logger in `utils.file_logger.py`.
*   This system logs detailed information, including file content (for non-binary files) or metadata (for images/binary), to `logs/file_creation_log.json`.
*   This JSON log is distinct from the main application logging in `logs/app.log` and provides a structured audit trail for file manipulations.

**Viewing Logs**:
*   Logs are output to the console (typically INFO level and above by default).
*   Detailed logs (typically DEBUG level and above by default) are stored in `logs/app.log`.
*   The `file_creation_log.json` in the `logs/` directory contains the audit trail for file operations.

## Directory overview

- `run.py` – Console runner that starts the agent and sampling loop.
- `agent.py` – Core agent implementation communicating with the LLM and tools.
- `prompts/` – Text files describing tasks that can be selected at startup.
- `tools/` – Implementation of Bash, project setup, code generation and other tools.
- `system_prompt/` – Base system prompt and code prompt helpers.
- `utils/` – Helper modules for displays, logging and Docker interaction.
- `logs/` – Log files and cached constants generated during runs.

## Notes

Some tools expect Docker to be available (container name defaults to
`python-dev-container`). Ensure the container is running if you plan to use tools
that execute commands inside Docker.

The Bash tool automatically converts commands to PowerShell when running on
Windows to keep things cross-platform.

All environment variables can be placed in a `.env` file. The main ones are
`OPENROUTER_API_KEY`/`OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`.

