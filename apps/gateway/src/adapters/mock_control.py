from __future__ import annotations

import json
from typing import Any


MOCK_METADATA_KEY = "respawn_mock"


def mock_options(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    raw = metadata.get(MOCK_METADATA_KEY)
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def mock_metadata(**options: Any) -> dict[str, str]:
    return {MOCK_METADATA_KEY: json.dumps(options, separators=(",", ":"))}
