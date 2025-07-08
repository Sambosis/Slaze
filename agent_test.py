import pytest
import types
import json
import os

from unittest.mock import MagicMock, patch, mock_open, AsyncMock

from agent import Agent

# Dummy classes to mock dependencies
class DummyDisplay:
    def __init__(self):
        self.messages = []
    def add_message(self, role, content):
        self.messages.append((role, content))
    async def wait_for_user_input(self, prompt):
        return "continue"

class DummyToolResult:
    def __init__(self, output=None, tool_name=None, error=None, base64_image=None):
        self.output = output
        self.tool_name = tool_name
        self.error = error
        self.base64_image = base64_image

class DummyToolCollection:
    def __init__(self, *args, **kwargs):
        pass
    def to_params(self):
        return [{"name": "dummy_tool"}]
    async def run(self, name, tool_input):
        return DummyToolResult(output="ok", tool_name=name)

class DummyOutputManager:
    def __init__(self, display):
        pass

class DummyOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace()
        self.chat.completions = types.SimpleNamespace()
        self.chat.completions.create = MagicMock()

@pytest.fixture(autouse=True)
def patch_agent_deps(monkeypatch):
    # Patch all external dependencies in Agent
    monkeypatch.setattr("agent.ToolCollection", DummyToolCollection)
    monkeypatch.setattr("agent.OutputManager", DummyOutputManager)
    monkeypatch.setattr("agent.OpenAI", lambda *a, **k: DummyOpenAIClient())
    monkeypatch.setattr("agent.reload_system_prompt", lambda: "system prompt")
    monkeypatch.setattr("agent.extract_text_from_content", lambda x: str(x))
    monkeypatch.setattr("agent.refresh_context_async", AsyncMock(return_value="refreshed context"))
    monkeypatch.setattr("agent.MAIN_MODEL", "gpt-4")
    monkeypatch.setattr("agent.MAX_SUMMARY_TOKENS", 100)
    monkeypatch.setattr("agent.COMPUTER_USE_BETA_FLAG", True)
    monkeypatch.setattr("agent.PROMPT_CACHING_BETA_FLAG", False)

@pytest.mark.parametrize(
    "task,display_type",
    [
        ("do something", DummyDisplay),
        ("another task", DummyDisplay),
    ],
    ids=["basic-init", "different-task"]
)
def test_agent_init(task, display_type):
    # Arrange

    # Act
    agent = Agent(task, display_type(), manual_tool_confirmation=False)

    # Assert
    assert agent.task == task
    assert isinstance(agent.display, display_type)
    assert agent.system_prompt == "system prompt"
    assert agent.messages[0]["content"] == "system prompt"
    assert agent.tool_params == [{"name": "dummy_tool"}]
    assert agent.enable_prompt_caching is True
    assert agent.betas == [True, False]
    assert agent.image_truncation_threshold == 1
    assert agent.only_n_most_recent_images == 2
    assert agent.step_count == 0

@pytest.mark.parametrize(
    "combined_content,tool_name,tool_input,expected_lines,case_id",
    [
        (
            [
                {"type": "tool_result", "tool_use_id": "id1", "is_error": False, "content": "result"},
                {"type": "text", "text": "some text"}
            ],
            "toolA",
            {"foo": "bar"},
            ["TOOL EXECUTION: toolA", "INPUT: {\n  \"foo\": \"bar\"\n}", "CONTENT TYPE: tool_result", "TOOL USE ID: id1", "ERROR: False", "CONTENT: result", "CONTENT TYPE: text", "TEXT:\nsome text"],
            "happy-path"
        ),
        (
            [
                {"type": "tool_result", "tool_use_id": "id2", "is_error": True, "content": [{"type": "text", "text": "err"}]},
                {"type": "text", "text": "other text"}
            ],
            "toolB",
            {"baz": 123},
            ["TOOL EXECUTION: toolB", "INPUT: {\n  \"baz\": 123\n}", "CONTENT TYPE: tool_result", "TOOL USE ID: id2", "ERROR: True", "CONTENT:", "  - text: err", "CONTENT TYPE: text", "TEXT:\nother text"],
            "list-content"
        ),
        (
            [
                {"type": "tool_result", "tool_use_id": "id3", "is_error": False, "content": None},
                {"type": "text", "text": "empty content"}
            ],
            "toolC",
            {},
            ["TOOL EXECUTION: toolC", "INPUT: {}", "CONTENT TYPE: tool_result", "TOOL USE ID: id3", "ERROR: False", "CONTENT: None", "CONTENT TYPE: text", "TEXT:\nempty content"],
            "none-content"
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None
)
def test_log_tool_results(tmp_path, combined_content, tool_name, tool_input, expected_lines, case_id):
    # Arrange
    agent = Agent("task", DummyDisplay(), manual_tool_confirmation=False)
    log_file = tmp_path / "tool.txt"
    os.makedirs(tmp_path, exist_ok=True)
    # Patch open to write to tmp_path
    with patch("builtins.open", mock_open()) as m:
        # Act
        agent.log_tool_results(combined_content, tool_name, tool_input)
        # Assert
        handle = m()
        written = "".join(call.args[0] for call in handle.write.call_args_list)
        for line in expected_lines:
            assert line in written

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_result,content_block,expected_is_error,expected_output,case_id",
    [
        (DummyToolResult(output="ok", tool_name="tool", error=None), {"name": "tool", "id": "tid", "input": {}}, False, "ok", "success"),
        (DummyToolResult(output=None, tool_name="tool", error="fail"), {"name": "tool", "id": "tid", "input": {}}, True, "fail", "error-attr"),
        (DummyToolResult(output=None, tool_name="tool", error=None, base64_image="imgdata"), {"name": "tool", "id": "tid", "input": {}}, False, None, "image"),
        (None, {"name": "tool", "id": "tid", "input": {}}, True, None, "none-result"),
        ("string error", {"name": "tool", "id": "tid", "input": {}}, True, "string error", "string-result"),
    ],
    ids=lambda x: x if isinstance(x, str) else None
)
async def test_make_api_tool_result(tool_result, content_block, expected_is_error, expected_output, case_id):
    # Arrange
    agent = Agent("task", DummyDisplay(), manual_tool_confirmation=False)

    # Act
    result = agent._make_api_tool_result(tool_result, content_block["id"])

    # Assert
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == content_block["id"]
    assert result["is_error"] == expected_is_error
    if expected_output is not None:
        assert any(expected_output in c.get("text", "") for c in result["content"])
    if tool_result and getattr(tool_result, "base64_image", None):
        assert any(c.get("type") == "image" for c in result["content"])

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_run_result,run_raises,expected_output,expected_error,case_id",
    [
        (DummyToolResult(output="ok", tool_name="tool"), False, "ok", None, "happy"),
        (None, False, "Tool execution failed with no result", None, "none-result"),
        (Exception("fail"), True, "Tool execution failed: fail", "fail", "exception"),
    ],
    ids=lambda x: x if isinstance(x, str) else None
)
async def test_run_tool(tool_run_result, run_raises, expected_output, expected_error, case_id):
    # Arrange
    agent = Agent("task", DummyDisplay(), manual_tool_confirmation=False)
    content_block = {"name": "tool", "id": "tid", "input": {"foo": "bar"}}
    if run_raises:
        async def raise_exc(*a, **k): raise Exception("fail")
        agent.tool_collection.run = raise_exc
    else:
        async def run(*a, **k): return tool_run_result
        agent.tool_collection.run = run

    with patch.object(agent, "log_tool_results") as log_mock:
        # Act
        result = await agent.run_tool(content_block)

        # Assert
        assert result["type"] == "tool_result"
        if expected_error:
            assert any(expected_error in c.get("text", "") for c in result["content"])
        else:
            assert any(expected_output in c.get("text", "") for c in result["content"])
        log_mock.assert_called_once()

@pytest.mark.parametrize(
    "messages,expected_breakpoints,case_id",
    [
        (
            [
                {"role": "user", "content": [{"foo": 1}, {"bar": 2}]},
                {"role": "user", "content": [{"foo": 3}, {"bar": 4}]},
                {"role": "assistant", "content": "hi"},
            ],
            0,
            "two-user-messages"
        ),
        (
            [
                {"role": "user", "content": [{"foo": 1}]},
                {"role": "assistant", "content": "hi"},
            ],
            1,
            "one-user-message"
        ),
        (
            [
                {"role": "assistant", "content": "hi"},
            ],
            2,
            "no-user-message"
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else None
)
def test_inject_prompt_caching(messages, expected_breakpoints, case_id):
    # Arrange
    agent = Agent("task", DummyDisplay(), manual_tool_confirmation=False)
    agent.messages = messages.copy()

    # Act
    agent._inject_prompt_caching()

    # Assert
    # Check that cache_control is set for up to 2 user messages with list content
    count = 0
    for m in reversed(agent.messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            if "cache_control" in m["content"][-1]:
                count += 1
    assert count <= 2

@pytest.mark.parametrize(
    "name,expected,case_id",
    [
        ("abcDEF-123_", "abcDEF-123_", "valid"),
        ("abc!@#def", "abc___def", "special-chars"),
        ("a"*70, "a"*64, "truncate"),
        ("", "", "empty"),
    ],
    ids=lambda x: x if isinstance(x, str) else None
)
def test_sanitize_tool_name(name, expected, case_id):
    # Arrange
    agent = Agent("task", DummyDisplay(), manual_tool_confirmation=False)

    # Act
    result = agent._sanitize_tool_name(name)

    # Assert
    assert result == expected

@pytest.mark.asyncio
async def test_step_happy(monkeypatch):
    # Arrange
    agent = Agent("task", DummyDisplay(), manual_tool_confirmation=False)
    agent.messages = [{"role": "user", "content": "hi"}]
    fake_response = types.SimpleNamespace()
    fake_msg = types.SimpleNamespace()
    fake_msg.content = "assistant says"
    fake_msg.tool_calls = None
    fake_response.choices = [types.SimpleNamespace(message=fake_msg)]
    agent.client.chat.completions.create = MagicMock(return_value=fake_response)

    # Act
    result = await agent.step()

    # Assert
    assert result is True
    assert agent.messages[-1]["role"] == "assistant"
    assert agent.messages[-1]["content"] == "assistant says"

@pytest.mark.asyncio
async def test_step_llm_error(monkeypatch):
    # Arrange
    agent = Agent("task", DummyDisplay(), manual_tool_confirmation=False)
    agent.messages = [{"role": "user", "content": "hi"}]
    agent.client.chat.completions.create = MagicMock(side_effect=Exception("fail"))
    agent.display = DummyDisplay()

    # Act
    result = await agent.step()

    # Assert
    assert result is True
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[0]["content"] == "refreshed context"
    assert agent.context_recently_refreshed is True

@pytest.mark.asyncio
async def test_step_with_tool_calls(monkeypatch):
    # Arrange
    agent = Agent("task", DummyDisplay(), manual_tool_confirmation=False)
    agent.messages = [{"role": "user", "content": "hi"}]
    fake_tc = types.SimpleNamespace()
    fake_tc.function = types.SimpleNamespace()
    fake_tc.function.name = "tool"
    fake_tc.function.arguments = json.dumps({"foo": "bar"})
    fake_tc.id = "tid"
    fake_response = types.SimpleNamespace()
    fake_msg = types.SimpleNamespace()
    fake_msg.content = None
    fake_msg.tool_calls = [fake_tc]
    fake_response.choices = [types.SimpleNamespace(message=fake_msg)]
    agent.client.chat.completions.create = MagicMock(return_value=fake_response)
    agent.run_tool = AsyncMock(return_value={
        "type": "tool_result",
        "content": [{"type": "text", "text": "result"}],
        "tool_use_id": "tid",
        "is_error": False,
    })

    # Act
    result = await agent.step()

    # Assert
    assert result is True
    assert agent.messages[-1]["role"] == "tool"
    assert agent.messages[-1]["tool_call_id"] == "tid"
    assert "result" in agent.messages[-1]["content"]

@pytest.mark.asyncio
async def test_step_with_tool_calls_no_content(monkeypatch):
    # Arrange
    agent = Agent("task", DummyDisplay(), manual_tool_confirmation=False)
    agent.messages = [{"role": "user", "content": "hi"}]
    fake_tc = types.SimpleNamespace()
    fake_tc.function = types.SimpleNamespace()
    fake_tc.function.name = "tool"
    fake_tc.function.arguments = None
    fake_tc.id = "tid"
    fake_response = types.SimpleNamespace()
    fake_msg = types.SimpleNamespace()
    fake_msg.content = None
    fake_msg.tool_calls = [fake_tc]
    fake_response.choices = [types.SimpleNamespace(message=fake_msg)]
    agent.client.chat.completions.create = MagicMock(return_value=fake_response)
    agent.run_tool = AsyncMock(return_value={
        "type": "tool_result",
        "content": [{"type": "text", "text": "result"}],
        "tool_use_id": "tid",
        "is_error": False,
    })

    # Act
    result = await agent.step()

    # Assert
    assert result is True
    assert agent.messages[-1]["role"] == "tool"
    assert agent.messages[-1]["tool_call_id"] == "tid"
    assert "result" in agent.messages[-1]["content"]

@pytest.mark.asyncio
async def test_step_no_tool_calls_user_exit(monkeypatch):
    # Arrange
    agent = Agent("task", DummyDisplay(), manual_tool_confirmation=False)
    agent.messages = [{"role": "user", "content": "hi"}]
    fake_response = types.SimpleNamespace()
    fake_msg = types.SimpleNamespace()
    fake_msg.content = "assistant says"
    fake_msg.tool_calls = None
    fake_response.choices = [types.SimpleNamespace(message=fake_msg)]
    agent.client.chat.completions.create = MagicMock(return_value=fake_response)
    agent.display = DummyDisplay()
    agent.display.wait_for_user_input = AsyncMock(return_value="exit")

    # Act
    result = await agent.step()

    # Assert
    assert result is False

@pytest.mark.asyncio
async def test_step_no_tool_calls_user_continue(monkeypatch):
    # Arrange
    agent = Agent("task", DummyDisplay(), manual_tool_confirmation=False)
    agent.messages = [{"role": "user", "content": "hi"}]
    fake_response = types.SimpleNamespace()
    fake_msg = types.SimpleNamespace()
    fake_msg.content = "assistant says"
    fake_msg.tool_calls = None
    fake_response.choices = [types.SimpleNamespace(message=fake_msg)]
    agent.client.chat.completions.create = MagicMock(return_value=fake_response)
    agent.display = DummyDisplay()
    agent.display.wait_for_user_input = AsyncMock(return_value="keep going")

    # Act
    result = await agent.step()

    # Assert
    assert result is True
    assert agent.messages[-1]["role"] == "user"
    assert agent.messages[-1]["content"] == "keep going"
