import json
import types
from utils.context_helpers import format_messages_to_string


def test_format_messages_to_string_basic():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]
    result = format_messages_to_string(messages)
    assert "USER:" in result
    assert "hello" in result
    assert "ASSISTANT:" in result
    assert "hi" in result


def test_format_messages_with_tool_calls_and_results():
    tc = types.SimpleNamespace()
    tc.function = types.SimpleNamespace(name="echo", arguments=json.dumps({"a": 1}))
    tc.id = "tid"
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [tc]},
        {
            "role": "tool",
            "tool_call_id": "tid",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tid",
                    "is_error": False,
                    "content": [{"type": "text", "text": "done"}],
                }
            ],
        },
    ]
    result = format_messages_to_string(messages)
    assert "Tool Call -> echo" in result
    assert "Arguments" in result
    assert "Tool Call ID: tid" in result
    assert "Tool Result" in result
    assert "done" in result
