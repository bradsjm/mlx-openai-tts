from __future__ import annotations

import os

from mlx_openai_tts.models.base import ModelAdapter
from mlx_openai_tts.registry import ModelSpec

_ALLOWED_EXTENSIONS = {".wav", ".flac"}


class ChatterboxAdapter(ModelAdapter):
    def __init__(self, spec: ModelSpec, *, voice_clone_dir: str | None) -> None:
        super().__init__(spec)
        self._voice_clone_dir = voice_clone_dir

    @property
    def requires_voice(self) -> bool:
        return False

    def resolve_voice(self, requested_voice: str | None) -> str | None:
        if requested_voice is None:
            return None
        value = requested_voice.strip()
        if not value:
            raise RuntimeError("voice must be non-empty")
        if value.lower() == "default":
            return None
        return self._resolve_ref_audio(value)

    def _resolve_ref_audio(self, voice: str) -> str:
        value = voice.strip()
        if not value:
            raise RuntimeError("voice must be non-empty")
        if value in {".", ".."}:
            raise RuntimeError("voice must be a filename, not a path")
        if os.path.basename(value) != value:
            raise RuntimeError("voice must be a filename, not a path")
        if os.path.sep in value or (os.path.altsep and os.path.altsep in value):
            raise RuntimeError("voice must be a filename, not a path")
        voice_clone_dir = self._voice_clone_dir
        if not voice_clone_dir:
            raise RuntimeError("Voice cloning requires TTS_VOICE_CLONE_DIR to be set")
        if not os.path.isdir(voice_clone_dir):
            raise RuntimeError(
                f"Voice clone directory {voice_clone_dir!r} does not exist or is not a directory"
            )
        root, ext = os.path.splitext(value)
        if ext:
            if ext.lower() not in _ALLOWED_EXTENSIONS:
                allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
                raise RuntimeError(f"Unsupported voice extension {ext!r}. Allowed: {allowed}")
            return self._find_voice_file(voice_clone_dir, value)
        for suffix in (".wav", ".flac"):
            candidate = f"{root}{suffix}"
            resolved = self._find_voice_file(voice_clone_dir, candidate, raise_if_missing=False)
            if resolved is not None:
                return resolved
        raise RuntimeError(f"Voice file {value!r} not found in {voice_clone_dir!r}")

    def _find_voice_file(
        self,
        voice_clone_dir: str,
        filename: str,
        *,
        raise_if_missing: bool = True,
    ) -> str | None:
        candidate = os.path.join(voice_clone_dir, filename)
        if os.path.isfile(candidate):
            return candidate
        try:
            entries = os.listdir(voice_clone_dir)
        except OSError as exc:
            raise RuntimeError(f"Failed to list voice clone directory {voice_clone_dir!r}") from exc
        target = filename.lower()
        for entry in entries:
            if entry.lower() == target:
                resolved = os.path.join(voice_clone_dir, entry)
                if os.path.isfile(resolved):
                    return resolved
        if raise_if_missing:
            raise RuntimeError(f"Voice file {filename!r} not found in {voice_clone_dir!r}")
        return None

    def _build_generate_kwargs(
        self,
        *,
        text: str,
        voice: str | None,
        speed: float | None,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {"text": text}
        if speed is not None and self._supports_generate_param("speed"):
            kwargs["speed"] = float(speed)
        if voice is not None:
            if self._supports_generate_param("ref_audio"):
                kwargs["ref_audio"] = voice
            elif self._supports_generate_param("audio_prompt_path"):
                kwargs["audio_prompt_path"] = voice
            elif self._supports_generate_param("audio_prompt"):
                kwargs["audio_prompt"] = voice
            else:
                raise RuntimeError("MLX chatterbox model does not accept reference audio inputs")
        return kwargs
