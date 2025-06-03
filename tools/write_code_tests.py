import pytest
import types
import builtins
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Import the function under test
from tools.write_code import WriteCodeTool
from tools.write_code import LLMResponseError, APIError

class DummyDisplay:
    def __init__(self):
        self.messages = []
    def add_message(self, role, msg):
        self.messages.append((role, msg))

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "display_present, all_skeletons, external_imports, internal_imports, file_path, task_file_exists, task_file_content, llm_result, exception_to_raise, expected_in_result, test_id",
    [
        # Happy path: display present, no task.txt, no exception
        (
            True,
            {"foo.py": "def foo(): pass"},
            ["os"],
            ["my_module"],
            Path("foo.py"),
            False,
            "",
            "def foo():\n    pass",
            None,
            "def foo():\n    pass",
            "happy-path-display"
        ),
        # Happy path: display absent, task.txt present, no exception
        (
            False,
            {"bar.py": "def bar(): pass"},
            [],
            [],
            Path("bar.py"),
            True,
            "Task from file",
            "def bar():\n    pass",
            None,
            "def bar():\n    pass",
            "happy-path-no-display-task-file"
        ),
        # Edge: empty skeletons, empty imports, no task.txt
        (
            True,
            {},
            [],
            [],
            Path("empty.py"),
            False,
            "",
            "# generated code",
            None,
            "# generated code",
            "edge-empty-skeletons-imports"
        ),
        # Error: LLMResponseError
        (
            True,
            {"baz.py": "def baz(): pass"},
            [],
            [],
            Path("baz.py"),
            False,
            "",
            None,
            "LLMResponseError",
            "# Error generating code for baz.py: LLMResponseError - Simulated LLM error",
            "error-llmresponse"
        ),
        # Error: APIError
        (
            True,
            {"qux.py": "def qux(): pass"},
            [],
            [],
            Path("qux.py"),
            False,
            "",
            None,
            "APIError",
            "# Error generating code for qux.py: API Error - Simulated API error",
            "error-apierror"
        ),
        # Error: generic Exception
        (
            True,
            {"err.py": "def err(): pass"},
            [],
            [],
            Path("err.py"),
            False,
            "",
            None,
            "Exception",
            "# Error generating code for err.py (final): Simulated generic error",
            "error-generic-exception"
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None
)
async def test_call_llm_to_generate_code(
    display_present, all_skeletons, external_imports, internal_imports, file_path,
    task_file_exists, task_file_content, llm_result, exception_to_raise, expected_in_result, test_id
):
    # Arrange

    # Patch os.path.exists and open for task.txt logic
    patches = [
        patch("c:\\Users\\Machine81\\Slazy\\tools\\write_code.os.path.exists", return_value=task_file_exists),
        patch("c:\\Users\\Machine81\\Slazy\\tools\\write_code.open", create=True),
        patch("c:\\Users\\Machine81\\Slazy\\tools\\write_code.get_constant", side_effect=lambda k: None),
        patch("c:\\Users\\Machine81\\Slazy\\tools\\write_code.code_prompt_generate", side_effect=lambda **kwargs: "PROMPT"),
        patch("c:\\Users\\Machine81\\Slazy\\tools\\write_code.ic"),
        patch("c:\\Users\\Machine81\\Slazy\\tools\\write_code.rr"),
    ]
    for p in patches:
        p.start()

    # Patch open to return task_file_content if needed
    if task_file_exists:
        mock_file = MagicMock()
        mock_file.read.return_value = task_file_content
        patches[1].return_value.__enter__.return_value = mock_file

    # Patch _llm_generate_code_core_with_retry to simulate LLM or raise exceptions
    class DummySelf:
        def __init__(self):
            self.display = DummyDisplay() if display_present else None
            self._log_generated_output = MagicMock()
        async def _llm_generate_code_core_with_retry(self, **kwargs):
            if exception_to_raise == "LLMResponseError":
                raise type("LLMResponseError", (Exception,), {})( "Simulated LLM error")
            if exception_to_raise == "APIError":
                raise type("APIError", (Exception,), {})( "Simulated API error")
            if exception_to_raise == "Exception":
                raise Exception("Simulated generic error")
            return llm_result

    dummy_self = DummySelf()

    # Act
    result = await _call_llm_to_generate_code(
        dummy_self,
        code_description="desc",
        all_skeletons=all_skeletons,
        external_imports=external_imports,
        internal_imports=internal_imports,
        file_path=file_path,
    )

    # Assert
    assert expected_in_result in result

    # Clean up patches
    for p in patches:
        p.stop()
