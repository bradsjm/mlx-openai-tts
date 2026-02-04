from __future__ import annotations

from mlx_openai_tts.models.base import ModelAdapter
from mlx_openai_tts.models.chatterbox import ChatterboxAdapter
from mlx_openai_tts.models.kokoro import KokoroAdapter
from mlx_openai_tts.registry import ModelSpec


def create_adapter(*, spec: ModelSpec, voice_clone_dir: str | None) -> ModelAdapter:
    if spec.model_type == "kokoro":
        return KokoroAdapter(spec)
    if spec.model_type == "chatterbox":
        return ChatterboxAdapter(spec, voice_clone_dir=voice_clone_dir)
    raise RuntimeError(f"Unsupported model_type {spec.model_type!r}")
