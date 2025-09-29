import asyncio
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_stub)

flask_stub = types.ModuleType("flask")
flask_stub.g = types.SimpleNamespace()
flask_stub.Flask = lambda *args, **kwargs: types.SimpleNamespace(
    route=lambda *r_args, **r_kwargs: (lambda func: func),
    config={},
)
flask_stub.render_template = lambda *args, **kwargs: ""
flask_stub.jsonify = lambda *args, **kwargs: {}
flask_stub.request = types.SimpleNamespace(form={}, args={}, json=None, files={})
flask_stub.send_file = lambda *args, **kwargs: None
sys.modules.setdefault("flask", flask_stub)

openai_stub = types.ModuleType("openai")
sys.modules.setdefault("openai", openai_stub)

socketio_stub = types.ModuleType("flask_socketio")


class _DummySocketIO:
    def __init__(self, *args, **kwargs):
        pass

    def start_background_task(self, target, *args, **kwargs):
        return target(*args, **kwargs)

    def emit(self, *args, **kwargs):
        pass

    def on(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


socketio_stub.SocketIO = _DummySocketIO
sys.modules.setdefault("flask_socketio", socketio_stub)

portalocker_stub = types.ModuleType("portalocker")


class _DummyLock:
    def __init__(self, filename, mode="r", timeout=0, **_):
        self._fp = open(filename, mode)

    def __enter__(self):
        return self._fp

    def __exit__(self, exc_type, exc, tb):
        self._fp.close()
        return False


portalocker_stub.Lock = _DummyLock
sys.modules.setdefault("portalocker", portalocker_stub)

rich_stub = types.ModuleType("rich")
rich_stub.__path__ = []
rich_console_stub = types.ModuleType("rich.console")
rich_panel_stub = types.ModuleType("rich.panel")
rich_prompt_stub = types.ModuleType("rich.prompt")
rich_syntax_stub = types.ModuleType("rich.syntax")


class _DummyConsole:
    def print(self, *args, **kwargs):
        pass


class _DummyPanel:
    def __init__(self, *args, **kwargs):
        self.renderable = args[0] if args else None


class _DummyPrompt:
    @staticmethod
    def ask(*args, default="", **kwargs):
        return default


class _DummyConfirm:
    @staticmethod
    def ask(*args, default=False, **kwargs):
        return default


class _DummyIntPrompt:
    @staticmethod
    def ask(*args, choices=None, default=0, **kwargs):
        if default:
            return default
        if choices:
            return choices[0]
        return 0


class _DummySyntax:
    def __init__(self, code: str, *args, **kwargs):
        self.code = code


rich_console_stub.Console = _DummyConsole
rich_panel_stub.Panel = _DummyPanel
rich_prompt_stub.Prompt = _DummyPrompt
rich_prompt_stub.Confirm = _DummyConfirm
rich_prompt_stub.IntPrompt = _DummyIntPrompt
rich_syntax_stub.Syntax = _DummySyntax
rich_stub.console = rich_console_stub
sys.modules.setdefault("rich", rich_stub)
sys.modules.setdefault("rich.console", rich_console_stub)
sys.modules.setdefault("rich.panel", rich_panel_stub)
sys.modules.setdefault("rich.prompt", rich_prompt_stub)
sys.modules.setdefault("rich.syntax", rich_syntax_stub)


aiohttp_stub = types.ModuleType("aiohttp")


class _DummyClientTimeout:
    def __init__(self, total=None):
        self.total = total


class _DummyResponse:
    status = 200

    async def text(self):
        return ""

    async def json(self):
        return {"choices": [{"message": {"content": ""}}]}


class _DummyPostContext:
    async def __aenter__(self):
        return _DummyResponse()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyClientSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        return _DummyPostContext()


aiohttp_stub.ClientSession = _DummyClientSession
aiohttp_stub.ClientTimeout = _DummyClientTimeout
sys.modules.setdefault("aiohttp", aiohttp_stub)

from config import get_constant, set_constant

TOOLS_DIR = ROOT / "tools"
tools_pkg = types.ModuleType("tools")
tools_pkg.__path__ = [str(TOOLS_DIR)]
sys.modules.setdefault("tools", tools_pkg)

spec = importlib.util.spec_from_file_location(
    "tools.edit_llm", TOOLS_DIR / "edit_llm.py", submodule_search_locations=[str(TOOLS_DIR)]
)
edit_llm_module = importlib.util.module_from_spec(spec)
sys.modules["tools.edit_llm"] = edit_llm_module
spec.loader.exec_module(edit_llm_module)  # type: ignore[arg-type]

EditLLMTool = edit_llm_module.EditLLMTool


class _DummyLLMClient:
    def __init__(self, response: str):
        self._response = response
        self.called = False

    async def call(self, *args, **kwargs):
        self.called = True
        return self._response


@pytest.fixture()
def repo_dir(tmp_path: Path):
    original_repo = get_constant("REPO_DIR")
    original_api_key = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    set_constant("REPO_DIR", tmp_path)
    try:
        yield tmp_path
    finally:
        set_constant("REPO_DIR", original_repo)
        if original_api_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = original_api_key


def test_llm_str_replace_prefers_traditional_for_code(repo_dir: Path):
    target_file = repo_dir / "snippet.py"
    target_file.write_text("value = 1\n", encoding="utf-8")

    tool = EditLLMTool()
    dummy_client = _DummyLLMClient("value = 1\n")
    tool._llm_client = dummy_client

    result = asyncio.run(tool(
        command="str_replace",
        path=str(target_file),
        old_str="value = 1",
        new_str="value = 2",
    ))

    assert result.error is None, f"Unexpected error: {result.error}"
    assert not dummy_client.called, "LLM should not be invoked for simple snippet replacement"
    assert target_file.read_text(encoding="utf-8") == "value = 2\n"


def test_llm_str_replace_uses_llm_for_natural_language(repo_dir: Path):
    target_file = repo_dir / "function.py"
    target_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

    desired_output = "def foo():\n    return 2\n"
    tool = EditLLMTool()
    dummy_client = _DummyLLMClient(desired_output)
    tool._llm_client = dummy_client

    result = asyncio.run(tool(
        command="str_replace",
        path=str(target_file),
        old_str="the return statement in foo",
        new_str="Change the return value to 2",
    ))

    assert result.error is None, f"Unexpected error: {result.error}"
    assert dummy_client.called, "LLM should handle natural language instructions"
    assert target_file.read_text(encoding="utf-8").rstrip("\n") == desired_output.rstrip("\n")
