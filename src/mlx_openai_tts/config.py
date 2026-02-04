from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODELS_JSON = "models.json"
DEFAULT_MAX_CHARS = 4096
DEFAULT_WARMUP_TEXT = "Hello from MLX TTS."


def _env_int(name: str, default: int) -> int:
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
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_float(name: str, default: float) -> float:
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
    api_key: str | None
    models_path: str
    max_chars: int
    warmup_text: str
    strict: bool
    voice_clone_dir: str | None


def load_config() -> AppConfig:
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
