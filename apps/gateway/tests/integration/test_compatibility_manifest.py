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


def test_responses_compatibility_manifest_is_machine_readable(client):
    response = client.get("/compatibility/responses")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "respawn.responses_compatibility_manifest"
    assert body["version"]
    assert body["source"] == "docs/RESPONSES_COMPATIBILITY.md"
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
