"""Adapter for Kokoro TTS model.

Kokoro is a neural TTS model that requires voice selection
and supports speed adjustment.
"""

from __future__ import annotations

from mlx_openai_tts.models.base import ModelAdapter


class KokoroAdapter(ModelAdapter):
    """Adapter for Kokoro TTS models.

    Kokoro requires a voice parameter and supports optional speed control.
    Voice must be selected from the configured voices list.
    """

    @property
    def requires_voice(self) -> bool:
        """Kokoro models require a voice parameter."""
        return True

    def resolve_voice(self, requested_voice: str | None) -> str:
        """Resolve voice identifier for Kokoro model.

        Args:
            requested_voice: Voice ID from request, or None to use default.

        Returns:
            Resolved voice identifier.

        Raises:
            RuntimeError: If voice is missing, empty, or not in allowed list.
        """
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
        """Build kwargs for Kokoro model.generate().

        Args:
            text: Input text to synthesize.
            voice: Resolved voice identifier.
            speed: Playback speed multiplier.

        Returns:
            Dictionary of parameters for Kokoro model.

        Raises:
            RuntimeError: If voice is None or model doesn't support required params.
        """
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
