import json
from pathlib import Path
import pytest
from utils.web_ui import WebUI
from config import set_constant

@pytest.fixture
def client(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("hello")
    (repo / "sub").mkdir()
    (repo / "sub" / "b.txt").write_text("world")
    set_constant("REPO_DIR", repo)
    ui = WebUI(lambda *a, **k: None)
    ui.app.testing = True
    return ui.app.test_client()


def test_file_tree_endpoint(client):
    resp = client.get('/api/file_tree')
    assert resp.status_code == 200
    files = json.loads(resp.data)
    assert 'a.txt' in files
    assert 'sub/b.txt' in files


def test_get_file_content(client):
    resp = client.get('/api/file?path=a.txt')
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data['content'] == 'hello'


