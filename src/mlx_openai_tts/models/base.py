"""Base model adapter for MLX TTS backends.

Defines the interface for model adapters that wrap mlx-audio
TTS models, providing voice resolution and audio generation.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Protocol

import numpy as np

from mlx_openai_tts.audio import ArrayLike, coerce_audio_1d_float32
from mlx_openai_tts.registry import ModelSpec

try:
    from mlx_audio.tts.utils import load_model
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "mlx-audio import failed. Activate your venv and install requirements."
    ) from exc


class GenerateResult(Protocol):
    """Protocol for MLX model generation result."""

    audio: ArrayLike


class TtsModel(Protocol):
    """Protocol for MLX TTS model interface."""

    def generate(self, **kwargs: object) -> Iterable[GenerateResult]:
        """Generate audio chunks from text.

        Args:
            **kwargs: Model-specific generation parameters.

        Yields:
            GenerateResult objects containing audio chunks.
        """


class ModelAdapter:
    """Base adapter for MLX TTS models.

    Provides common functionality for model loading, parameter inspection,
    and audio generation. Subclasses must implement voice resolution and
    parameter building.

    Attributes:
        spec: Model specification.
        model: Loaded MLX TTS model instance.
        sample_rate: Audio sample rate in Hz.
        _generate_params: Set of parameter names supported by model.generate().
    """

    def __init__(self, spec: ModelSpec) -> None:
        """Initialize adapter with model specification.

        Args:
            spec: Model specification from registry.
        """
        self.spec = spec
        self.model: TtsModel | None = None
        self.sample_rate = 24_000
        self._generate_params: set[str] = set()

    @property
    def requires_voice(self) -> bool:
        """Whether this model requires a voice parameter.

        Returns:
            True if voice is required, False otherwise.
        """
        raise NotImplementedError

    def resolve_voice(self, requested_voice: str | None) -> str | None:
        """Resolve voice identifier for model.

        Args:
            requested_voice: Voice ID from request, or None.

        Returns:
            Resolved voice identifier for the model.

        Raises:
            RuntimeError: If voice resolution fails.
        """
        raise NotImplementedError

    def load(self, *, strict: bool) -> None:
        """Load the MLX model and inspect its parameters.

        Args:
            strict: Whether to use strict mode for model loading.

        Raises:
            RuntimeError: If model loading or inspection fails.
        """
        kwargs: dict[str, bool] = {}
        try:
            sig = inspect.signature(load_model)
            if "strict" in sig.parameters:
                kwargs["strict"] = strict
        except (TypeError, ValueError):
            pass
        try:
            model = load_model(self.spec.repo_id, **kwargs)
        except ValueError as exc:
            raise RuntimeError(
                "Failed to load MLX model weights. This usually means the model weights "
                "don't match the installed mlx-audio version. Try setting "
                "TTS_MLX_STRICT=false or switch to a compatible MLX model."
            ) from exc
        self.model = model
        try:
            sig = inspect.signature(model.generate)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Failed to inspect MLX model generate signature.") from exc
        self._generate_params = set(sig.parameters)
        try:
            rate = model.sample_rate
        except AttributeError:
            rate = None
        if isinstance(rate, (int, float)) and rate > 0:
            self.sample_rate = int(rate)

    def iter_chunks(
        self,
        *,
        text: str,
        voice: str | None,
        speed: float | None,
    ) -> Iterable[np.ndarray]:
        """Generate audio as an iterable of chunks.

        Args:
            text: Input text to synthesize.
            voice: Resolved voice identifier.
            speed: Playback speed multiplier.

        Yields:
            Float32 mono audio chunks as numpy arrays.

        Raises:
            RuntimeError: If model not loaded or generation fails.
        """
        model = self.model
        if model is None:
            raise RuntimeError("MLX engine not loaded.")
        kwargs = self._build_generate_kwargs(text=text, voice=voice, speed=speed)
        for result in model.generate(**kwargs):
            try:
                audio = result.audio
            except AttributeError:
                raise RuntimeError("MLX generate result missing audio attribute.") from None
            if audio is None:
                continue
            yield coerce_audio_1d_float32(audio)

    def synthesize_full(
        self,
        *,
        text: str,
        voice: str | None,
        speed: float | None,
    ) -> np.ndarray:
        """Generate complete audio as a single array.

        Args:
            text: Input text to synthesize.
            voice: Resolved voice identifier.
            speed: Playback speed multiplier.

        Returns:
            Float32 mono audio as numpy array.

        Raises:
            RuntimeError: If no audio is produced.
        """
        chunks = list(self.iter_chunks(text=text, voice=voice, speed=speed))
        if not chunks:
            raise RuntimeError("MLX pipeline produced no audio. Check voice configuration.")
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def _supports_generate_param(self, name: str) -> bool:
        """Check if model.generate() accepts a parameter.

        Args:
            name: Parameter name to check.

        Returns:
            True if parameter is supported, False otherwise.
        """
        return name in self._generate_params

    def _build_generate_kwargs(
        self,
        *,
        text: str,
        voice: str | None,
        speed: float | None,
    ) -> dict[str, object]:
        """Build kwargs for model.generate() based on model capabilities.

        Args:
            text: Input text to synthesize.
            voice: Resolved voice identifier (or None).
            speed: Playback speed multiplier (or None).

        Returns:
            Dictionary of parameters for model.generate().

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError
