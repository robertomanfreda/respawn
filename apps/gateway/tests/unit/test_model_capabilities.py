from src.config import Settings
from src.services.model_capabilities import reasoning_efforts_for_model, reasoning_supported_for_model


def test_reasoning_efforts_are_explicit_per_model():
    settings = Settings(
        model_capabilities=(
            "model-a=text,reasoning,reasoning-effort-low,reasoning-effort-high;"
            "model-b=text,reasoning;"
            "model-c=text,reasoning-effort-low"
        )
    )

    assert reasoning_efforts_for_model("model-a", settings) == {"low", "high"}
    assert reasoning_supported_for_model("model-a", settings) is True
    assert reasoning_efforts_for_model("model-b", settings) == set()
    assert reasoning_supported_for_model("model-b", settings) is True
    assert reasoning_efforts_for_model("model-c", settings) == {"low"}
    assert reasoning_supported_for_model("model-c", settings) is False


def test_reasoning_effort_capability_matches_base_model_alias():
    settings = Settings(model_capabilities="gpt-oss=text,reasoning,reasoning-effort-low")

    assert reasoning_efforts_for_model("gpt-oss:120b", settings) == {"low"}
