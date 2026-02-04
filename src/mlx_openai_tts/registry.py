"""Model registry configuration and validation.

Loads and validates model specifications from models.json, providing
a resolved registry with models indexed by ID.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

ModelType = Literal["kokoro", "chatterbox"]


class ModelSpec(BaseModel):
    id: str
    repo_id: str
    model_type: ModelType = Field(default="kokoro")
    voices: list[str] = Field(default_factory=list)
    default_voice: str | None = None
    warmup_text: str | None = None


class ModelRegistry(BaseModel):
    default_model: str | None = None
    models: list[ModelSpec]


@dataclass(frozen=True)
class ResolvedRegistry:
    registry: ModelRegistry
    models_by_id: dict[str, ModelSpec]
    default_model: str


def load_registry_payload(payload: Mapping[str, object]) -> ResolvedRegistry:
    try:
        registry = ModelRegistry.model_validate(payload)
    except Exception as exc:
        raise RuntimeError(f"Invalid models JSON schema: {exc}") from exc

    if not registry.models:
        raise RuntimeError("Models JSON must contain at least one model")

    models_by_id: dict[str, ModelSpec] = {}
    for spec in registry.models:
        model_id = spec.id.strip()
        if not model_id:
            raise RuntimeError("Model id must be non-empty")
        if model_id in models_by_id:
            raise RuntimeError(f"Duplicate model id {model_id!r} in models JSON")
        if not spec.repo_id.strip():
            raise RuntimeError(f"Model {model_id!r} repo_id must be non-empty")
        voices = [voice.strip() for voice in spec.voices if voice.strip()]
        spec.voices = voices
        if voices and spec.default_voice is None:
            spec.default_voice = voices[0]
        if spec.default_voice is not None and voices and spec.default_voice not in voices:
            raise RuntimeError(
                f"Model {model_id!r} default_voice {spec.default_voice!r} not in voices list"
            )
        models_by_id[model_id] = spec

    default_model = registry.default_model or registry.models[0].id
    if default_model not in models_by_id:
        raise RuntimeError(f"default_model {default_model!r} not found in models list")

    return ResolvedRegistry(
        registry=registry, models_by_id=models_by_id, default_model=default_model
    )


def load_registry(path: str) -> ResolvedRegistry:
    if not os.path.exists(path):
        raise RuntimeError(f"Models JSON not found at {path!r}")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read models JSON at {path!r}: {exc}") from exc
    return load_registry_payload(payload)
