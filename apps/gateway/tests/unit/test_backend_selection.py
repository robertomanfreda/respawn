from types import SimpleNamespace

import pytest

from src.adapters.ollama_backend import OllamaBackend
from src.config import Settings
from src.main import build_backend


def test_settings_default_backend_is_ollama(monkeypatch):
    monkeypatch.delenv("MODEL_BACKEND", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)

    settings = Settings()

    assert settings.model_backend == "ollama"
    assert settings.default_model == "gpt-oss:120b"


def test_build_backend_supports_ollama():
    backend = build_backend(
        SimpleNamespace(
            model_backend="ollama",
            ollama_base_url="http://ollama.test/v1",
            backend_timeout_seconds=10,
        )
    )

    assert isinstance(backend, OllamaBackend)
    assert backend.base_url == "http://ollama.test/v1"


def test_build_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported MODEL_BACKEND"):
        build_backend(
            SimpleNamespace(
                model_backend="ollmaa",
                default_model="gpt-oss-120b",
            )
        )
