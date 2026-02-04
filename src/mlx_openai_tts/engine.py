from __future__ import annotations

import logging
import threading
from collections.abc import Iterable

import numpy as np

from .config import AppConfig
from .models import ModelAdapter, create_adapter
from .registry import ModelSpec, ResolvedRegistry


class MlxTtsEngine:
    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter
        self.sample_rate = adapter.sample_rate

    def load(self, *, strict: bool) -> None:
        self.adapter.load(strict=strict)
        self.sample_rate = self.adapter.sample_rate

    def iter_chunks(
        self,
        *,
        text: str,
        voice: str | None,
        speed: float | None,
    ) -> Iterable[np.ndarray]:
        return self.adapter.iter_chunks(text=text, voice=voice, speed=speed)

    def synthesize_full(
        self,
        *,
        text: str,
        voice: str | None,
        speed: float | None,
    ) -> np.ndarray:
        return self.adapter.synthesize_full(text=text, voice=voice, speed=speed)


class ModelManager:
    def __init__(
        self,
        *,
        registry: ResolvedRegistry,
        infer_lock: threading.Lock,
        strict: bool,
        cfg: AppConfig,
    ):
        self.registry = registry
        self.infer_lock = infer_lock
        self.strict = strict
        self.cfg = cfg
        self.active_key: str | None = None
        self.active_spec: ModelSpec | None = None
        self.active_engine: MlxTtsEngine | None = None
        self.active_adapter: ModelAdapter | None = None
        self._engines_by_id: dict[str, tuple[MlxTtsEngine, ModelAdapter]] = {}

    def load_default(self, *, warmup_text: str) -> None:
        spec = self.registry.models_by_id[self.registry.default_model]
        engine, adapter = self._get_cached_engine(spec)
        self.active_key = spec.id
        self.active_spec = spec
        self.active_engine = engine
        self.active_adapter = adapter
        voice = None
        if spec.default_voice:
            voice = adapter.resolve_voice(spec.default_voice)
        text = spec.warmup_text or warmup_text
        if text and (voice is not None or not adapter.requires_voice):
            engine.synthesize_full(text=text, voice=voice, speed=1.0)

    def load_all(self) -> None:
        logger = logging.getLogger("uvicorn.error")
        for spec in self.registry.registry.models:
            if spec.id in self._engines_by_id:
                continue
            logger.info("Preloading model %s from %s", spec.id, spec.repo_id)
            try:
                engine, adapter = self._load_engine(spec)
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                raise RuntimeError(f"Failed to preload model {spec.id!r}: {exc}") from exc
            self._engines_by_id[spec.id] = (engine, adapter)

    def _load_engine(self, spec: ModelSpec) -> tuple[MlxTtsEngine, ModelAdapter]:
        adapter = create_adapter(spec=spec, voice_clone_dir=self.cfg.voice_clone_dir)
        engine = MlxTtsEngine(adapter)
        engine.load(strict=self.strict)
        return engine, adapter

    def _get_cached_engine(self, spec: ModelSpec) -> tuple[MlxTtsEngine, ModelAdapter]:
        cached = self._engines_by_id.get(spec.id)
        if cached is not None:
            return cached
        engine, adapter = self._load_engine(spec)
        self._engines_by_id[spec.id] = (engine, adapter)
        return engine, adapter

    def get_engine(self, model_key: str) -> tuple[MlxTtsEngine, ModelSpec, ModelAdapter]:
        if (
            self.active_key == model_key
            and self.active_engine
            and self.active_spec
            and self.active_adapter
        ):
            return self.active_engine, self.active_spec, self.active_adapter
        spec = self.registry.models_by_id.get(model_key)
        if spec is None:
            raise RuntimeError(f"Invalid model {model_key!r}")
        with self.infer_lock:
            if (
                self.active_key == model_key
                and self.active_engine
                and self.active_spec
                and self.active_adapter
            ):
                return self.active_engine, self.active_spec, self.active_adapter
            engine, adapter = self._get_cached_engine(spec)
            self.active_key = spec.id
            self.active_spec = spec
            self.active_engine = engine
            self.active_adapter = adapter
        return engine, spec, adapter

    def resolve_voice(self, model_key: str, requested_voice: str | None) -> str | None:
        _engine, spec, adapter = self.get_engine(model_key)
        return adapter.resolve_voice(requested_voice)
