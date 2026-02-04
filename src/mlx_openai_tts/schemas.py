"""Pydantic schemas for OpenAI-compatible TTS API.

Defines request/response models for the TTS endpoints, matching
OpenAI's audio/speech API specification.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ResponseFormat = Literal["wav", "mp3", "opus", "pcm", "aac", "flac"]
StreamFormat = Literal["audio", "sse"]


class VoicePayload(BaseModel):
    """Voice identifier payload for flexible voice specification."""

    id: str


class AudioSpeechRequest(BaseModel):
    """Request model for audio speech generation endpoint.

    Compatible with OpenAI's /v1/audio/speech API.

    Attributes:
        model: Model identifier to use for synthesis.
        input: Text to synthesize (required, min 1 char).
        voice: Voice identifier or VoicePayload.
        instructions: Additional synthesis instructions.
        response_format: Output audio format (default: mp3).
        stream_format: Streaming mode: 'audio' or 'sse' (default: audio).
        speed: Playback speed multiplier, 0.25-4.0 (default: 1.0).
    """

    model: str | None = Field(default=None)
    input: str = Field(min_length=1)
    voice: str | VoicePayload | None = None
    instructions: str | None = None
    response_format: ResponseFormat = Field(default="mp3")
    stream_format: StreamFormat = Field(default="audio")
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)


PermissionEntry = dict[str, str | int | bool | None]


class ModelListItem(BaseModel):
    """Individual model entry in model list response.

    Attributes:
        id: Model identifier.
        object: Object type (always "model").
        owned_by: Owner identifier (always "local").
        permission: List of permission entries (empty for local).
    """

    id: str
    object: str
    owned_by: str
    permission: list[PermissionEntry]


class ModelListResponse(BaseModel):
    """Response model for /v1/models endpoint.

    Attributes:
        object: Object type (always "list").
        data: List of available models.
        default_model: Default model identifier.
        default_voice: Default voice identifier for default model.
    """

    object: str
    data: list[ModelListItem]
    default_model: str
    default_voice: str | None


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Health status ("ok").
        name: Service name.
        version: Service version.
        active_model: Currently loaded model identifier.
        repo_id: Active model's repository ID.
        default_voice: Default voice for active model.
        sample_rate: Audio sample rate in Hz.
    """

    status: str
    name: str
    version: str
    active_model: str | None
    repo_id: str | None
    default_voice: str | None
    sample_rate: int | None
