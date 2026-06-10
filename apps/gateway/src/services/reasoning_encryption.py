import base64
import hashlib
import hmac
import json
import secrets
from typing import Any


ENVELOPE_VERSION = 1
ENVELOPE_ALGORITHM = "respawn-hmac-sha256-stream"


def seal_reasoning_content(
    reasoning_text: str,
    *,
    key: str,
    response_id: str,
    item_id: str,
) -> str:
    key_bytes = _key_bytes(key)
    nonce = secrets.token_bytes(16)
    plaintext = json.dumps(
        {
            "reasoning": reasoning_text,
            "response_id": response_id,
            "item_id": item_id,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    ciphertext = _xor(plaintext, _keystream(key_bytes, nonce, len(plaintext)))
    envelope = {
        "v": ENVELOPE_VERSION,
        "alg": ENVELOPE_ALGORITHM,
        "kid": key_id(key),
        "nonce": _b64url(nonce),
        "ciphertext": _b64url(ciphertext),
    }
    tag_payload = _tag_payload(envelope)
    envelope["tag"] = _b64url(hmac.new(key_bytes, tag_payload, hashlib.sha256).digest())
    return _b64url(json.dumps(envelope, separators=(",", ":")).encode("utf-8"))


def unseal_reasoning_content(blob: str, *, key: str) -> dict[str, Any]:
    key_bytes = _key_bytes(key)
    try:
        envelope = json.loads(_b64url_decode(blob))
        tag = _b64url_decode(str(envelope.pop("tag")))
        expected = hmac.new(key_bytes, _tag_payload(envelope), hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("reasoning encrypted_content tag mismatch")
        nonce = _b64url_decode(str(envelope["nonce"]))
        ciphertext = _b64url_decode(str(envelope["ciphertext"]))
        plaintext = _xor(ciphertext, _keystream(key_bytes, nonce, len(ciphertext)))
        return json.loads(plaintext)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid reasoning encrypted_content envelope") from exc


def key_id(key: str) -> str:
    return hashlib.sha256(_key_bytes(key)).hexdigest()[:16]


def _key_bytes(key: str) -> bytes:
    if key.startswith("base64:"):
        return base64.b64decode(key.removeprefix("base64:"))
    return key.encode("utf-8")


def _tag_payload(envelope: dict[str, Any]) -> bytes:
    return json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _keystream(key: bytes, nonce: bytes, size: int) -> bytes:
    stream = bytearray()
    counter = 0
    while len(stream) < size:
        stream.extend(hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(stream[:size])


def _xor(left: bytes, right: bytes) -> bytes:
    return bytes(left_byte ^ right_byte for left_byte, right_byte in zip(left, right, strict=True))


def _b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _b64url_decode(payload: str) -> bytes:
    padding = "=" * (-len(payload) % 4)
    return base64.urlsafe_b64decode(payload + padding)
