from __future__ import annotations

import hmac

from fastapi import HTTPException


def require_auth(authorization: str | None, api_key: str | None) -> None:
    if api_key is None:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token, api_key):
        raise HTTPException(status_code=401, detail="Invalid token")
