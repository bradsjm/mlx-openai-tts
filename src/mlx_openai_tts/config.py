"""Configuration management for the TTS server.

Loads configuration from environment variables with validation and defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODELS_JSON = "models.json"
DEFAULT_MAX_CHARS = 4096
DEFAULT_WARMUP_TEXT = "Hello from MLX TTS."


def _env_int(name: str, default: int) -> int:
    """Parse an environment variable as an integer.

    Args:
        name: Environment variable name.
        default: Default value if variable is not set.

    Returns:
        The parsed integer value.

    Raises:
        RuntimeError: If the value is not a valid integer or is <= 0.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name}={raw!r}, expected integer") from exc
    if value <= 0:
        raise RuntimeError(f"Invalid {name}={raw!r}, expected > 0")
    return value


def _env_str(name: str, default: str) -> str:
    """Parse an environment variable as a string.

    Args:
        name: Environment variable name.
        default: Default value if variable is not set or empty.

    Returns:
        The trimmed string value or default.
    """
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_float(name: str, default: float) -> float:
    """Parse an environment variable as a float.

    Args:
        name: Environment variable name.
        default: Default value if variable is not set.

    Returns:
        The parsed float value.

    Raises:
        RuntimeError: If the value is not a valid float or is <= 0.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name}={raw!r}, expected float") from exc
    if value <= 0:
        raise RuntimeError(f"Invalid {name}={raw!r}, expected > 0")
    return value


@dataclass(frozen=True)
class AppConfig:
    """Immutable configuration for the TTS server.

    Attributes:
        api_key: API key for authentication, or None to disable auth.
        models_path: Path to the models.json configuration file.
        max_chars: Maximum input text length in characters.
        warmup_text: Text to synthesize for model warmup on startup.
        strict: Whether to use strict mode for model loading.
        voice_clone_dir: Directory containing voice reference audio files.
    """

    api_key: str | None
    models_path: str
    max_chars: int
    warmup_text: str
    strict: bool
    voice_clone_dir: str | None


def load_config() -> AppConfig:
    """Load configuration from environment variables.

    Reads and validates environment variables for all configuration options.

    Returns:
        The loaded application configuration.
    """
    strict_env = _env_str("TTS_MLX_STRICT", "false").lower()
    strict = strict_env in {"1", "true", "yes", "on"}
    voice_clone_dir = _env_str("TTS_VOICE_CLONE_DIR", "").strip()
    return AppConfig(
        api_key=os.getenv("API_KEY") or None,
        models_path=_env_str("TTS_MODELS_JSON", DEFAULT_MODELS_JSON),
        max_chars=_env_int("TTS_MAX_CHARS", DEFAULT_MAX_CHARS),
        warmup_text=_env_str("TTS_WARMUP_TEXT", DEFAULT_WARMUP_TEXT),
        strict=strict,
        voice_clone_dir=voice_clone_dir or None,
    )
