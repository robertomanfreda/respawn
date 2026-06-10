import os

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


@pytest.fixture
def client(tmp_path):
    previous_env = {key: os.environ.get(key) for key in ["DATABASE_URL", "MODEL_BACKEND", "AUTH_DISABLED", "DEFAULT_MODEL", "PROMPT_CACHE_MIN_TOKENS", *WEB_SEARCH_ENV, *IMAGE_GENERATION_ENV]}
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    os.environ["MODEL_BACKEND"] = "mock"
    os.environ["AUTH_DISABLED"] = "true"
    os.environ["DEFAULT_MODEL"] = "gpt-oss-120b"
    os.environ["PROMPT_CACHE_MIN_TOKENS"] = "8"
    os.environ["WEB_SEARCH_ENABLED"] = "false"
    os.environ["IMAGE_GENERATION_ENABLED"] = "false"
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
def web_search_client(tmp_path):
    previous_env = {key: os.environ.get(key) for key in ["DATABASE_URL", "MODEL_BACKEND", "AUTH_DISABLED", "DEFAULT_MODEL", "PROMPT_CACHE_MIN_TOKENS", *WEB_SEARCH_ENV, *IMAGE_GENERATION_ENV]}
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path / 'web_search.db'}"
    os.environ["MODEL_BACKEND"] = "mock"
    os.environ["AUTH_DISABLED"] = "true"
    os.environ["DEFAULT_MODEL"] = "gpt-oss-120b"
    os.environ["PROMPT_CACHE_MIN_TOKENS"] = "8"
    os.environ["WEB_SEARCH_ENABLED"] = "true"
    os.environ["WEB_SEARCH_BACKEND"] = "mock"
    os.environ["WEB_SEARCH_MAX_RESULTS"] = "5"
    os.environ["IMAGE_GENERATION_ENABLED"] = "false"
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
def image_generation_client(tmp_path):
    previous_env = {key: os.environ.get(key) for key in ["DATABASE_URL", "MODEL_BACKEND", "AUTH_DISABLED", "DEFAULT_MODEL", "PROMPT_CACHE_MIN_TOKENS", *WEB_SEARCH_ENV, *IMAGE_GENERATION_ENV]}
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path / 'image_generation.db'}"
    os.environ["MODEL_BACKEND"] = "mock"
    os.environ["AUTH_DISABLED"] = "true"
    os.environ["DEFAULT_MODEL"] = "gpt-oss-120b"
    os.environ["PROMPT_CACHE_MIN_TOKENS"] = "8"
    os.environ["WEB_SEARCH_ENABLED"] = "false"
    os.environ["IMAGE_GENERATION_ENABLED"] = "true"
    os.environ["IMAGE_GENERATION_BACKEND"] = "mock"
    os.environ["IMAGE_GENERATION_DEFAULT_SIZE"] = "512x512"
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
