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
    audio: ArrayLike


class TtsModel(Protocol):
    def generate(self, **kwargs: object) -> Iterable[GenerateResult]: ...


class ModelAdapter:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.model: TtsModel | None = None
        self.sample_rate = 24_000
        self._generate_params: set[str] = set()

    @property
    def requires_voice(self) -> bool:
        raise NotImplementedError

    def resolve_voice(self, requested_voice: str | None) -> str | None:
        raise NotImplementedError

    def load(self, *, strict: bool) -> None:
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
        chunks = list(self.iter_chunks(text=text, voice=voice, speed=speed))
        if not chunks:
            raise RuntimeError("MLX pipeline produced no audio. Check voice configuration.")
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    def _supports_generate_param(self, name: str) -> bool:
        return name in self._generate_params

    def _build_generate_kwargs(
        self,
        *,
        text: str,
        voice: str | None,
        speed: float | None,
    ) -> dict[str, object]:
        raise NotImplementedError
