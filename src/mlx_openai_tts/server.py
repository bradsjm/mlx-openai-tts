"""FastAPI server for OpenAI-compatible TTS API.

Provides REST endpoints for text-to-speech generation with model
management, authentication, and streaming support.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse

from . import __version__
from .audio import build_full_response, stream_pcm_audio, stream_sse
from .auth import require_auth
from .config import AppConfig, _env_int, _env_str, load_config
from .engine import ModelManager
from .registry import ResolvedRegistry, load_registry
from .schemas import (
    AudioSpeechRequest,
    HealthResponse,
    ModelListItem,
    ModelListResponse,
    VoicePayload,
)


def _normalize_text(text: str) -> str:
    """Normalize whitespace in input text."""
    return re.sub(r"\s+", " ", text).strip()


def _parse_voice_id(voice: str | VoicePayload) -> str:
    """Parse voice identifier from string or VoicePayload.

    Args:
        voice: Voice as string or VoicePayload.

    Returns:
        Parsed voice identifier.

    Raises:
        RuntimeError: If voice is empty or invalid.
    """
    if isinstance(voice, str):
        value = voice.strip()
        if not value:
            raise RuntimeError("voice must be non-empty")
        return value
    value = voice.id.strip()
    if value:
        return value
    raise RuntimeError("Invalid voice format; expected string or {id: str}")


def _configure_loguru() -> None:
    """Configure loguru to route logs to uvicorn."""
    try:
        from loguru import logger as loguru_logger
    except ImportError:
        return

    def sink(message: object) -> None:
        try:
            record = message.record["logging_record"]
        except (AttributeError, KeyError, TypeError):
            return
        logging.getLogger("uvicorn.error").handle(record)

    loguru_logger.remove()
    loguru_logger.add(sink, level="INFO")


APP_VERSION = __version__


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app.

    Loads configuration, models, and performs warmup on startup.

    Args:
        app: FastAPI application instance.

    Yields:
        None when startup is complete.

    Raises:
        RuntimeError: If model loading or warmup fails.
    """
    cfg = load_config()
    app.state.cfg = cfg
    logger = logging.getLogger("uvicorn.error")
    _configure_loguru()

    app.state.infer_lock = threading.Lock()
    registry = load_registry(cfg.models_path)
    app.state.registry = registry
    app.state.model_manager = ModelManager(
        registry=registry,
        infer_lock=app.state.infer_lock,
        strict=cfg.strict,
        cfg=cfg,
    )

    try:
        with app.state.infer_lock:
            logger.info("Preloading %d models", len(registry.registry.models))
            app.state.model_manager.load_all()
            logger.info("Model preload complete")
            app.state.model_manager.load_default(warmup_text=cfg.warmup_text)
    except Exception as exc:
        logger.exception("Startup failed while loading models")
        raise RuntimeError(
            "Startup failed while preloading or warming models. This typically indicates an "
            "MLX model for the installed mlx-audio version. Try switching to a "
            "different model in models.json or set TTS_MLX_STRICT=false."
        ) from exc

    yield


app = FastAPI(title="Local MLX TTS (OpenAI-compatible)", version=APP_VERSION, lifespan=lifespan)

logging.getLogger("phonemizer").setLevel(logging.ERROR)


def _require_auth(authorization: str | None) -> None:
    """Require valid authorization if API key is configured.

    Args:
        authorization: Authorization header value.

    Raises:
        HTTPException: If authentication fails.
    """
    cfg: AppConfig = app.state.cfg
    require_auth(authorization, cfg.api_key)


@app.get("/v1/models", response_model=ModelListResponse)
def list_models(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> ModelListResponse:
    """List available TTS models.

    Args:
        authorization: Authorization header value.

    Returns:
        Model list response with available models.
    """
    _require_auth(authorization)
    registry: ResolvedRegistry = app.state.registry
    data = [
        {"id": spec.id, "object": "model", "owned_by": "local", "permission": []}
        for spec in registry.registry.models
    ]
    return ModelListResponse(
        object="list",
        data=[ModelListItem(**item) for item in data],
        default_model=registry.default_model,
        default_voice=registry.models_by_id[registry.default_model].default_voice,
    )


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
@app.get("/v1", response_model=HealthResponse)
def health(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> HealthResponse:
    """Health check endpoint.

    Args:
        authorization: Authorization header value.

    Returns:
        Health response with server status and active model info.
    """
    _require_auth(authorization)
    manager: ModelManager = app.state.model_manager
    active_engine = manager.active_engine
    active_spec = manager.active_spec
    return HealthResponse(
        status="ok",
        name="mlx-openai-tts",
        version=APP_VERSION,
        active_model=active_spec.id if active_spec else None,
        repo_id=active_spec.repo_id if active_spec else None,
        default_voice=active_spec.default_voice if active_spec else None,
        sample_rate=active_engine.sample_rate if active_engine else None,
    )


@app.post("/v1/audio/speech")
async def audio_speech(
    body: AudioSpeechRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> Response:
    """Generate speech from text.

    Supports multiple response formats (wav, mp3, opus, pcm, aac, flac)
    and streaming modes (audio, sse).

    Args:
        body: Audio speech request with text, voice, and format options.
        authorization: Authorization header value.

    Returns:
        Streaming or complete audio response with metadata headers.

    Raises:
        HTTPException: If validation or synthesis fails.
    """
    _require_auth(authorization)
    cfg: AppConfig = app.state.cfg
    registry: ResolvedRegistry = app.state.registry
    model_manager: ModelManager = app.state.model_manager
    infer_lock: threading.Lock = app.state.infer_lock

    model_key = body.model or registry.default_model
    if model_key not in registry.models_by_id:
        raise HTTPException(status_code=400, detail=f"Invalid model {model_key!r}")
    model_spec = registry.models_by_id[model_key]

    text = _normalize_text(body.input)
    if not text:
        raise HTTPException(status_code=400, detail="input must be non-empty")
    if len(text) > cfg.max_chars:
        raise HTTPException(status_code=400, detail=f"input too long (>{cfg.max_chars} chars)")

    try:
        voice_id = _parse_voice_id(body.voice) if body.voice is not None else None
        voice = model_manager.resolve_voice(model_key, voice_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response_format = str(body.response_format).lower()
    stream_format = body.stream_format
    speed = body.speed if body.speed is not None else 1.0

    voice_label = voice_id or model_spec.default_voice or "default"
    headers = {
        "X-Model": model_key,
        "X-Voice": voice_label,
    }

    try:
        engine, _active_spec, _adapter = model_manager.get_engine(model_key)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to load model {model_key!r}: {exc}"
        ) from exc
    headers["X-Sample-Rate"] = str(engine.sample_rate)

    # Non-streaming or compressed format: synthesize full audio first
    if stream_format == "audio":
        if response_format != "pcm":
            t0 = time.perf_counter()
            try:
                with infer_lock:
                    audio_f32 = engine.synthesize_full(text=text, voice=voice, speed=speed)
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Synthesis failed for {model_key!r}: {exc}"
                ) from exc
            try:
                out_bytes, media_type = build_full_response(
                    audio_f32=audio_f32,
                    sample_rate=engine.sample_rate,
                    response_format=response_format,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            headers["X-Latency-Ms"] = str(int((time.perf_counter() - t0) * 1000))
            return StreamingResponse(iter([out_bytes]), media_type=media_type, headers=headers)

        # PCM streaming: stream audio chunks as they're generated
        try:
            audio_iter, latency_ms = stream_pcm_audio(
                engine=engine, infer_lock=infer_lock, text=text, voice=voice, speed=speed
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Synthesis failed for {model_key!r}: {exc}"
            ) from exc
        headers["X-Latency-Ms"] = str(latency_ms)
        return StreamingResponse(audio_iter, media_type="application/octet-stream", headers=headers)

    # SSE streaming: wrap audio chunks in server-sent events
    if stream_format == "sse":
        if response_format == "pcm":
            try:
                audio_iter, latency_ms = stream_pcm_audio(
                    engine=engine, infer_lock=infer_lock, text=text, voice=voice, speed=speed
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Synthesis failed for {model_key!r}: {exc}"
                ) from exc
            headers["X-Latency-Ms"] = str(latency_ms)
            sse_headers = {**headers, "Cache-Control": "no-cache"}
            return StreamingResponse(
                stream_sse(audio_bytes_iter=audio_iter),
                media_type="text/event-stream",
                headers=sse_headers,
            )

        # Non-PCM formats: synthesize full audio first, then stream as SSE
        t0 = time.perf_counter()
        try:
            with infer_lock:
                audio_f32 = engine.synthesize_full(text=text, voice=voice, speed=speed)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Synthesis failed for {model_key!r}: {exc}"
            ) from exc
        try:
            out_bytes, _media_type = build_full_response(
                audio_f32=audio_f32,
                sample_rate=engine.sample_rate,
                response_format=response_format,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        headers["X-Latency-Ms"] = str(int((time.perf_counter() - t0) * 1000))
        sse_headers = {**headers, "Cache-Control": "no-cache"}
        return StreamingResponse(
            stream_sse(audio_bytes_iter=[out_bytes]),
            media_type="text/event-stream",
            headers=sse_headers,
        )

    raise HTTPException(status_code=400, detail="Unsupported stream_format (use audio|sse)")


def main() -> None:
    """Run the TTS server.

    Loads configuration from environment variables and starts uvicorn.
    """
    host = _env_str("HOST", "0.0.0.0")
    port = _env_int("PORT", 8001)
    uvicorn.run("mlx_openai_tts.server:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
