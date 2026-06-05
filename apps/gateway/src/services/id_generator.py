import secrets


def generate_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(18).replace('-', '').replace('_', '')[:24]}"
