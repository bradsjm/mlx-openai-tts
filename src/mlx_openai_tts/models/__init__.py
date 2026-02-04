"""Model adapters for different TTS backends.

Provides factory function to create appropriate adapters based on
model_type (kokoro, chatterbox).
"""

from __future__ import annotations

from mlx_openai_tts.models.base import ModelAdapter
from mlx_openai_tts.models.chatterbox import ChatterboxAdapter
from mlx_openai_tts.models.kokoro import KokoroAdapter
from mlx_openai_tts.registry import ModelSpec


def create_adapter(*, spec: ModelSpec, voice_clone_dir: str | None) -> ModelAdapter:
    """Create a model adapter based on specification.

    Args:
        spec: Model specification containing model_type.
        voice_clone_dir: Directory for voice reference audio (chatterbox only).

    Returns:
        Appropriate ModelAdapter subclass instance.

    Raises:
        RuntimeError: If model_type is not supported.
    """
    if spec.model_type == "kokoro":
        return KokoroAdapter(spec)
    if spec.model_type == "chatterbox":
        return ChatterboxAdapter(spec, voice_clone_dir=voice_clone_dir)
    raise RuntimeError(f"Unsupported model_type {spec.model_type!r}")
