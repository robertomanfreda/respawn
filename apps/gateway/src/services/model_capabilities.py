from src.config import Settings


REASONING_EFFORT_CAPABILITY_PREFIX = "reasoning-effort-"


def model_capabilities(settings: Settings) -> dict[str, set[str]]:
    capabilities: dict[str, set[str]] = {}
    for entry in settings.model_capabilities.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        model, values = entry.split("=", 1)
        capabilities[model.strip().lower()] = {value.strip().lower() for value in values.split(",") if value.strip()}
    return capabilities


def capabilities_for_model(model: str, settings: Settings) -> set[str]:
    configured = model_capabilities(settings)
    model_key = model.lower()
    if model_key in configured:
        return configured[model_key]
    if ":" in model_key:
        base = model_key.split(":", 1)[0]
        if base in configured:
            return configured[base]
    return {"text"}


def reasoning_efforts_for_model(model: str, settings: Settings) -> set[str]:
    capabilities = capabilities_for_model(model, settings)
    return {
        capability.removeprefix(REASONING_EFFORT_CAPABILITY_PREFIX)
        for capability in capabilities
        if capability.startswith(REASONING_EFFORT_CAPABILITY_PREFIX)
    }


def reasoning_supported_for_model(model: str, settings: Settings) -> bool:
    return "reasoning" in capabilities_for_model(model, settings)
