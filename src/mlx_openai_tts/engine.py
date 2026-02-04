"""Engine and model management for TTS server.

Provides MlxTtsEngine wrapper for audio generation and ModelManager
for caching and switching between multiple models.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable

import numpy as np

from .config import AppConfig
from .models import ModelAdapter, create_adapter
from .registry import ModelSpec, ResolvedRegistry


class MlxTtsEngine:
    """Wrapper around ModelAdapter for audio generation.

    Delegates to the underlying model adapter and caches sample rate.

    Attributes:
        adapter: The underlying model adapter.
        sample_rate: Audio sample rate in Hz.
    """

    def __init__(self, adapter: ModelAdapter):
        """Initialize engine with model adapter.

        Args:
            adapter: Model adapter to wrap.
        """
        self.adapter = adapter
        self.sample_rate = adapter.sample_rate

    def load(self, *, strict: bool) -> None:
        """Load the underlying MLX model.

        Args:
            strict: Whether to use strict mode for model loading.
        """
        self.adapter.load(strict=strict)
        self.sample_rate = self.adapter.sample_rate

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

        Returns:
            Iterable of float32 mono audio chunks.
        """
        return self.adapter.iter_chunks(text=text, voice=voice, speed=speed)

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
        """
        return self.adapter.synthesize_full(text=text, voice=voice, speed=speed)


class ModelManager:
    """Manages multiple TTS models with caching and switching.

    Maintains a cache of loaded models and handles thread-safe
    switching between them. Uses double-checked locking pattern.

    Attributes:
        registry: Resolved model registry.
        infer_lock: Thread lock for serialization.
        strict: Whether to use strict mode for model loading.
        cfg: Application configuration.
        active_key: ID of currently active model.
        active_spec: Spec of currently active model.
        active_engine: Currently active engine.
        active_adapter: Currently active adapter.
        _engines_by_id: Cache of loaded engines by model ID.
    """

    def __init__(
        self,
        *,
        registry: ResolvedRegistry,
        infer_lock: threading.Lock,
        strict: bool,
        cfg: AppConfig,
    ):
        """Initialize model manager.

        Args:
            registry: Resolved model registry.
            infer_lock: Thread lock for inference serialization.
            strict: Whether to use strict mode for model loading.
            cfg: Application configuration.
        """
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
        """Load default model and perform warmup synthesis.

        Args:
            warmup_text: Text for warmup synthesis.
        """
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
        """Preload all models in the registry.

        Raises:
            RuntimeError: If any model fails to load.
        """
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
        """Load an engine and adapter for a model specification.

        Args:
            spec: Model specification.

        Returns:
            Tuple of (engine, adapter).
        """
        adapter = create_adapter(spec=spec, voice_clone_dir=self.cfg.voice_clone_dir)
        engine = MlxTtsEngine(adapter)
        engine.load(strict=self.strict)
        return engine, adapter

    def _get_cached_engine(self, spec: ModelSpec) -> tuple[MlxTtsEngine, ModelAdapter]:
        """Get cached engine or load if not present.

        Args:
            spec: Model specification.

        Returns:
            Tuple of (engine, adapter).
        """
        cached = self._engines_by_id.get(spec.id)
        if cached is not None:
            return cached
        engine, adapter = self._load_engine(spec)
        self._engines_by_id[spec.id] = (engine, adapter)
        return engine, adapter

    def get_engine(self, model_key: str) -> tuple[MlxTtsEngine, ModelSpec, ModelAdapter]:
        """Get engine for specified model, switching if necessary.

        Uses double-checked locking for thread-safe lazy loading.

        Args:
            model_key: Model identifier.

        Returns:
            Tuple of (engine, spec, adapter).

        Raises:
            RuntimeError: If model ID is invalid.
        """
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
        """Resolve voice identifier for a model.

        Args:
            model_key: Model identifier.
            requested_voice: Voice ID from request, or None.

        Returns:
            Resolved voice identifier.
        """
        _engine, spec, adapter = self.get_engine(model_key)
        return adapter.resolve_voice(requested_voice)
