from __future__ import annotations

from mlx_openai_tts.models.base import ModelAdapter


class KokoroAdapter(ModelAdapter):
    @property
    def requires_voice(self) -> bool:
        return True

    def resolve_voice(self, requested_voice: str | None) -> str:
        if requested_voice is None:
            if self.spec.default_voice is None:
                raise RuntimeError("voice is required for this model")
            return self.spec.default_voice
        voice_id = requested_voice.strip()
        if not voice_id:
            raise RuntimeError("voice must be non-empty")
        if self.spec.voices and voice_id not in self.spec.voices:
            allowed = ", ".join(sorted(self.spec.voices))
            raise RuntimeError(f"Unknown voice {voice_id!r}. Available: {allowed}")
        return voice_id

    def _build_generate_kwargs(
        self,
        *,
        text: str,
        voice: str | None,
        speed: float | None,
    ) -> dict[str, object]:
        if voice is None:
            raise RuntimeError("voice is required for this model")
        kwargs: dict[str, object] = {"text": text}
        if self._supports_generate_param("voice"):
            kwargs["voice"] = voice
        else:
            raise RuntimeError("MLX model does not accept voice inputs")
        if speed is not None and self._supports_generate_param("speed"):
            kwargs["speed"] = float(speed)
        return kwargs
