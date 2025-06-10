import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tools.write_code import WriteCodeTool, CodeCommand, FileDetail # Make sure this import path is correct
from config import get_constant, set_constant, TOP_LEVEL_DIR, LOGS_DIR # Ensure LOGS_DIR is available or set up

# Ensure LOGS_DIR exists for testing
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TEST_CONTEXT_LOG_FILE = LOGS_DIR / "test_llm_context_log.jsonl"

@pytest.fixture(autouse=True)
def manage_test_log_file():
    # Setup: Store original if exists, set to test log, ensure clean state
    original_log_file_path = get_constant("LLM_CONTEXT_LOG_FILE")
    set_constant("LLM_CONTEXT_LOG_FILE", str(TEST_CONTEXT_LOG_FILE))
    if TEST_CONTEXT_LOG_FILE.exists():
        TEST_CONTEXT_LOG_FILE.unlink()

    yield

    # Teardown: Clean up test log file, restore original
    if TEST_CONTEXT_LOG_FILE.exists():
        TEST_CONTEXT_LOG_FILE.unlink()
    if original_log_file_path:
        set_constant("LLM_CONTEXT_LOG_FILE", str(original_log_file_path))
    else:
        # If it wasn't set before, perhaps remove it or set to a default
        # For now, let's assume we just revert to its previous state (None or a value)
        pass


@pytest.mark.asyncio
async def test_llm_context_logging_for_skeleton_generation():
    # 1. Setup
    tool = WriteCodeTool()

    # Mock the actual LLM call to avoid external dependencies and control output
    # We want the logging part (which happens before the LLM call) to execute
    mock_llm_skeleton_call = AsyncMock(return_value="# Mocked skeleton content")

    # The path to the method to patch depends on its definition.
    # If _llm_generate_skeleton_core_with_retry is a method of WriteCodeTool:
    patch_path = "tools.write_code.WriteCodeTool._llm_generate_skeleton_core_with_retry"
    # If it's a module-level function in tools.write_code:
    # patch_path = "tools.write_code._llm_generate_skeleton_core_with_retry"

    with patch(patch_path, mock_llm_skeleton_call):
        # 2. Action
        file_details = [
            FileDetail(filename="test_file.py", code_description="A test python file")
        ]
        project_path = "test_project" # This will create repo/test_project

        # Ensure REPO_DIR is set for the tool's path logic if it relies on it.
        # This might be needed if set_project_dir or similar isn't called by the test runner.
        # For now, assume get_constant("REPO_DIR") works as expected.
        if not get_constant("REPO_DIR"):
            set_constant("REPO_DIR", str(TOP_LEVEL_DIR / "repo"))


        await tool(
            command=CodeCommand.WRITE_CODEBASE,
            files=[f.model_dump() for f in file_details],
            project_path=project_path
        )

    # 3. Assertions
    assert TEST_CONTEXT_LOG_FILE.exists(), "LLM context log file was not created."

    with open(TEST_CONTEXT_LOG_FILE, "r", encoding="utf-8") as f:
        log_lines = f.readlines()

    assert len(log_lines) >= 1, "LLM context log file is empty."

    found_log_entry = False
    for line in log_lines:
        try:
            log_entry = json.loads(line)
            if log_entry.get("target_file") == "test_file.py" and log_entry.get("type") == "skeleton_generation":
                assert "timestamp" in log_entry
                assert "context" in log_entry
                assert isinstance(log_entry["context"], list)
                # Check if the context contains the expected elements for skeleton prompt
                # This depends on the structure of code_skeleton_prompt
                # For example, check if any message in context has a role 'system' or 'user'
                assert any(msg.get("role") for msg in log_entry["context"]), "Context messages should have roles."
                found_log_entry = True
                break
        except json.JSONDecodeError:
            pytest.fail(f"Failed to parse log line as JSON: {line}")

    assert found_log_entry, "Expected log entry for 'test_file.py' (skeleton_generation) not found or is malformed."

# Add more tests for code_generation logging if time permits
