import os

import pytest
from fastapi.testclient import TestClient

from src.config import get_settings
from src.main import create_app


@pytest.fixture
def client(tmp_path):
    previous_env = {key: os.environ.get(key) for key in ["DATABASE_URL", "MODEL_BACKEND", "AUTH_DISABLED", "DEFAULT_MODEL", "PROMPT_CACHE_MIN_TOKENS"]}
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    os.environ["MODEL_BACKEND"] = "mock"
    os.environ["AUTH_DISABLED"] = "true"
    os.environ["DEFAULT_MODEL"] = "gpt-oss-120b"
    os.environ["PROMPT_CACHE_MIN_TOKENS"] = "8"
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
