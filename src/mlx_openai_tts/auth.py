"""Authentication utilities for the TTS server.

Provides Bearer token validation using constant-time comparison to prevent
timing attacks.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException


def require_auth(authorization: str | None, api_key: str | None) -> None:
    """Validate Bearer token authorization.

    If API key is None, authentication is disabled. Otherwise, validates
    the Bearer token using constant-time comparison.

    Args:
        authorization: The Authorization header value (e.g., "Bearer token").
        api_key: The expected API key, or None to disable auth.

    Raises:
        HTTPException: If authentication is enabled and token is missing or invalid.
    """
    if api_key is None:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token, api_key):
        raise HTTPException(status_code=401, detail="Invalid token")
