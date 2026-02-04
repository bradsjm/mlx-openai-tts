from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ResponseFormat = Literal["wav", "mp3", "opus", "pcm", "aac", "flac"]
StreamFormat = Literal["audio", "sse"]


class VoicePayload(BaseModel):
    id: str


class AudioSpeechRequest(BaseModel):
    model: str | None = Field(default=None)
    input: str = Field(min_length=1)
    voice: str | VoicePayload | None = None
    instructions: str | None = None
    response_format: ResponseFormat = Field(default="mp3")
    stream_format: StreamFormat = Field(default="audio")
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0)


PermissionEntry = dict[str, str | int | bool | None]


class ModelListItem(BaseModel):
    id: str
    object: str
    owned_by: str
    permission: list[PermissionEntry]


class ModelListResponse(BaseModel):
    object: str
    data: list[ModelListItem]
    default_model: str
    default_voice: str | None


class HealthResponse(BaseModel):
    status: str
    name: str
    version: str
    active_model: str | None
    repo_id: str | None
    default_voice: str | None
    sample_rate: int | None
