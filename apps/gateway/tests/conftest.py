import os
from contextlib import contextmanager
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.config import get_settings
from src.main import create_app


WEB_SEARCH_ENV = [
    "WEB_SEARCH_ENABLED",
    "WEB_SEARCH_BACKEND",
    "WEB_SEARCH_BASE_URL",
    "WEB_SEARCH_TIMEOUT_SECONDS",
    "WEB_SEARCH_MAX_RESULTS",
    "WEB_SEARCH_MAX_RESULT_CHARS",
    "WEB_SEARCH_ALLOWED_DOMAINS",
    "WEB_SEARCH_BLOCKED_DOMAINS",
]

IMAGE_GENERATION_ENV = [
    "IMAGE_GENERATION_ENABLED",
    "IMAGE_GENERATION_BACKEND",
    "IMAGE_GENERATION_BASE_URL",
    "IMAGE_GENERATION_MODEL",
    "IMAGE_GENERATION_TIMEOUT_SECONDS",
    "IMAGE_GENERATION_DEFAULT_SIZE",
    "IMAGE_GENERATION_MAX_PIXELS",
    "IMAGE_GENERATION_DEFAULT_STEPS",
    "IMAGE_GENERATION_MAX_STEPS",
    "IMAGE_GENERATION_DEFAULT_CFG_SCALE",
    "IMAGE_GENERATION_SAMPLER",
    "IMAGE_GENERATION_NEGATIVE_PROMPT",
    "IMAGE_GENERATION_OUTPUT_FORMAT",
]

TEST_ENV_KEYS = [
    "DATABASE_URL",
    "MODEL_BACKEND",
    "AUTH_DISABLED",
    "DEFAULT_MODEL",
    "PROMPT_CACHE_MIN_TOKENS",
    *WEB_SEARCH_ENV,
    *IMAGE_GENERATION_ENV,
]


def _base_env(tmp_path, database_name: str) -> dict[str, str]:
    return {
        "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / database_name}",
        "MODEL_BACKEND": "mock",
        "AUTH_DISABLED": "true",
        "DEFAULT_MODEL": "gpt-oss-120b",
        "PROMPT_CACHE_MIN_TOKENS": "8",
        "WEB_SEARCH_ENABLED": "false",
        "IMAGE_GENERATION_ENABLED": "false",
    }


@contextmanager
def _configured_client(tmp_path, database_name: str, **overrides: str) -> Iterator[TestClient]:
    previous_env = {key: os.environ.get(key) for key in TEST_ENV_KEYS}
    os.environ.update({**_base_env(tmp_path, database_name), **overrides})
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


@pytest.fixture
def client(tmp_path):
    with _configured_client(tmp_path, "test.db") as test_client:
        yield test_client


@pytest.fixture
def web_search_client(tmp_path):
    with _configured_client(
        tmp_path,
        "web_search.db",
        WEB_SEARCH_ENABLED="true",
        WEB_SEARCH_BACKEND="mock",
        WEB_SEARCH_MAX_RESULTS="5",
    ) as test_client:
        yield test_client


@pytest.fixture
def image_generation_client(tmp_path):
    with _configured_client(
        tmp_path,
        "image_generation.db",
        IMAGE_GENERATION_ENABLED="true",
        IMAGE_GENERATION_BACKEND="mock",
        IMAGE_GENERATION_DEFAULT_SIZE="512x512",
    ) as test_client:
        yield test_client


@pytest.fixture
def web_search_and_image_generation_client(tmp_path):
    with _configured_client(
        tmp_path,
        "web_search_image_generation.db",
        WEB_SEARCH_ENABLED="true",
        WEB_SEARCH_BACKEND="mock",
        WEB_SEARCH_MAX_RESULTS="5",
        IMAGE_GENERATION_ENABLED="true",
        IMAGE_GENERATION_BACKEND="mock",
        IMAGE_GENERATION_DEFAULT_SIZE="512x512",
    ) as test_client:
        yield test_client
