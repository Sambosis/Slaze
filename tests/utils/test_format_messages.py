from utils.context_helpers import format_messages_to_string


def test_format_messages_full():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "OK"},
                {"type": "tool_use", "name": "echo_tool", "input": {"text": "hello"}},
            ],
            "tool_calls": [
                {"id": "1", "function": {"name": "echo_tool", "arguments": '{"text":"hello"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "result text"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "1",
                    "is_error": False,
                    "content": [{"type": "text", "text": "done"}],
                }
            ],
        },
    ]

    output = format_messages_to_string(messages)

    assert "USER:" in output
    assert "hi" in output
    assert "TOOL CALL:" in output and "echo_tool" in output
    assert 'Args: {"text":"hello"}' in output
    assert "Tool Call: echo_tool" in output
    assert "Tool Result [ID: 1]" in output
    assert "done" in output
