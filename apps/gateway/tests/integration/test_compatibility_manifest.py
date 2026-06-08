def test_health_ready_include_safe_compatibility_metadata(client):
    health = client.get("/healthz")
    ready = client.get("/readyz")

    assert health.status_code == 200
    assert ready.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["version"]
    assert health.json()["compatibility_manifest_version"]
    assert ready.json()["status"] == "ready"
    assert ready.json()["model_backend"] == "mock"
    assert ready.json()["default_model"] == "gpt-oss-120b"
    assert ready.json()["compatibility_manifest_version"] == health.json()["compatibility_manifest_version"]
    checks = ready.json()["checks"]
    assert set(checks) == {"database", "ollama", "worker", "cache", "storage"}
    assert all(check["status"] == "ready" for check in checks.values())


def test_readyz_reports_database_cache_storage_and_worker_outages(client):
    class BrokenSession:
        async def __aenter__(self):
            raise RuntimeError("database unavailable")

        async def __aexit__(self, *_):
            return False

    original_session = client.app.state.async_session
    client.app.state.async_session = lambda: BrokenSession()
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["database"]["status"] == "not_ready"
    client.app.state.async_session = original_session

    original_cache = client.app.state.prompt_cache
    client.app.state.prompt_cache = None
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["cache"]["status"] == "not_ready"
    client.app.state.prompt_cache = original_cache

    original_storage_backend = client.app.state.settings.file_storage_backend
    client.app.state.settings.file_storage_backend = "broken"
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["storage"]["status"] == "not_ready"
    client.app.state.settings.file_storage_backend = original_storage_backend

    original_worker = client.app.state.platform_file_cleanup_task
    client.app.state.platform_file_cleanup_task = None
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["worker"]["status"] == "not_ready"
    client.app.state.platform_file_cleanup_task = original_worker


def test_responses_compatibility_manifest_is_machine_readable(client):
    response = client.get("/compatibility/responses")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "respawn.responses_compatibility_manifest"
    assert body["version"]
    assert body["source"] == "docs/COMPATIBILITY.md"
    assert body["summary"]["supported"] > 0
    assert body["summary"]["unsupported"] > 0
    feature_ids = {feature["id"] for feature in body["features"]}
    assert "endpoints.responses.create" in feature_ids
    assert "endpoints.responses.cancel" in feature_ids
    for feature in body["features"]:
        assert feature["id"]
        assert feature["category"]
        assert feature["surface"]
        assert feature["status"]
        assert isinstance(feature["tags"], list)
        if feature["status"].startswith("supported"):
            assert feature["benchmark_required"] is True
            assert feature["benchmark_case"]
