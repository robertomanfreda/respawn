from fastapi import Depends, Header

from src.config import Settings, get_settings
from src.schemas.errors import OpenAIError


async def tenant_id(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> str | None:
    if settings.auth_disabled:
        return None
    api_key = None
    if authorization and authorization.lower().startswith("bearer "):
        api_key = authorization.split(" ", 1)[1]
    tenant = settings.tenant_for_key(api_key)
    if tenant is None:
        raise OpenAIError("Invalid API key.", status_code=401, type="authentication_error", code="invalid_api_key")
    return tenant
