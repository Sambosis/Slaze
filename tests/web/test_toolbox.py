import pytest
from utils.web_ui import WebUI

@pytest.fixture
def client():
    ui = WebUI(lambda *a, **k: None)
    ui.app.testing = True
    return ui.app.test_client()


def test_tools_page(client):
    resp = client.get('/tools')
    assert resp.status_code == 200
    # basic check for at least one tool name in response
    assert b'bash' in resp.data


def test_run_bash_tool(client):
    resp = client.post('/tools/bash', data={'command': 'echo test'})
    assert resp.status_code == 200
    assert b'test' in resp.data
