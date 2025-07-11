# Slaze Web Interface Guide

This guide explains how to run the Slaze agent with the built-in web interface.

## Prerequisites

1. **Python**: Ensure Python 3.12 is installed. The specific version used for development is listed in `.python-version`.
2. **Dependencies**: Install the required packages with:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Variables**: At a minimum you need an API key for OpenAI or OpenRouter. Set them via environment variables or a `.env` file in the repository root. Example `.env`:
   ```
   OPENROUTER_API_KEY=your-key-here
   OPENAI_BASE_URL=https://openrouter.ai/api/v1  # optional
   ```

## Running the Web Interface

1. From the project root, start the web server:
   ```bash
   python run.py web
   ```
2. The server listens on port `5002` by default. The port can be changed with the `--port` option.
3. The CLI prints the address of the running server using your local IP. Open
   your browser to that address (for example `http://192.168.x.x:5002/`).
4. Use the interface to select an existing prompt or create a new one, then click **Start** to launch the agent.
5. The agent will stream its conversation to the page. Logs and generated files are stored under the `logs/` and `repo/` directories.

For command line usage, see the main [README](../README.md).
