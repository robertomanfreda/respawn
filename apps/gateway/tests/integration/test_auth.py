from fastapi.testclient import TestClient

from src.adapters.mock_control import mock_metadata
from src.config import get_settings
from src.main import create_app


def test_auth_enabled_scopes_responses_by_tenant(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'auth.db'}")
    monkeypatch.setenv("MODEL_BACKEND", "mock")
    monkeypatch.setenv("AUTH_DISABLED", "false")
    monkeypatch.setenv("LOCAL_OPENAI_API_KEYS", "key-a:tenant-a,key-b:tenant-b")
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        unauthorized = client.post("/v1/responses", json={"input": "secret"})
        assert unauthorized.status_code == 401

        created = client.post("/v1/responses", headers={"Authorization": "Bearer key-a"}, json={"input": "tenant scoped"}).json()
        assert client.get(f"/v1/responses/{created['id']}", headers={"Authorization": "Bearer key-a"}).status_code == 200
        assert client.get(f"/v1/responses/{created['id']}", headers={"Authorization": "Bearer key-b"}).status_code == 404
        assert client.get(f"/v1/responses/{created['id']}/input_items", headers={"Authorization": "Bearer key-a"}).status_code == 200
        assert client.get(f"/v1/responses/{created['id']}/input_items", headers={"Authorization": "Bearer key-b"}).status_code == 404
        background = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-a"},
            json={"input": "background tenant scoped", "background": True, "store": True, "metadata": mock_metadata(delay_seconds=0.15)},
        ).json()
        assert client.post(f"/v1/responses/{background['id']}/cancel", headers={"Authorization": "Bearer key-b"}).status_code == 404
        assert client.post(f"/v1/responses/{background['id']}/cancel", headers={"Authorization": "Bearer key-a"}).status_code == 200

        cross_tenant_child = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-b"},
            json={"previous_response_id": created["id"], "input": "continue"},
        )
        assert cross_tenant_child.status_code == 404
        assert cross_tenant_child.json()["error"]["param"] == "previous_response_id"

    get_settings.cache_clear()
