from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import time
from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from .config import _env_float

if TYPE_CHECKING:
    import threading

    from .engine import MlxTtsEngine

DEFAULT_FFMPEG_TIMEOUT = 30.0
ArrayLike = np.ndarray | Sequence[float] | Sequence[Sequence[float]]


def coerce_audio_1d_float32(audio: ArrayLike) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise RuntimeError(f"Unexpected audio shape {arr.shape!r}; expected mono 1-D.")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def _wav_bytes(audio_f32: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm16, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _pcm_s16le_bytes(audio_f32: np.ndarray) -> bytes:
    clipped = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes(order="C")


def _ffmpeg_transcode(*, wav_bytes: bytes, response_format: str) -> tuple[bytes, str]:
    timeout = _env_float("FFMPEG_TIMEOUT_SECONDS", DEFAULT_FFMPEG_TIMEOUT)
    ffmpeg_bin = (os.getenv("FFMPEG_BIN") or "").strip()
    if not ffmpeg_bin:
        ffmpeg_bin = shutil.which("ffmpeg") or ""
    if not ffmpeg_bin:
        for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
            if os.path.exists(candidate):
                ffmpeg_bin = candidate
                break
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg not found (required for mp3/opus/aac/flac). Install via `brew install ffmpeg`."
        )

    if response_format == "mp3":
        args = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "mp3",
            "pipe:1",
        ]
        media_type = "audio/mpeg"
    elif response_format == "opus":
        args = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            "48k",
            "-f",
            "ogg",
            "pipe:1",
        ]
        media_type = "audio/ogg"
    elif response_format == "aac":
        args = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "adts",
            "pipe:1",
        ]
        media_type = "audio/aac"
    elif response_format == "flac":
        args = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "flac",
            "pipe:1",
        ]
        media_type = "audio/flac"
    else:
        raise RuntimeError(f"Unsupported transcode format: {response_format!r}")

    try:
        proc = subprocess.run(
            args,
            input=wav_bytes,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg transcode timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg transcode failed: {exc.stderr.decode('utf-8', 'replace')[:500]}"
        ) from exc

    return proc.stdout, media_type


def _sse_event(payload: Mapping[str, object]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def build_full_response(
    *,
    audio_f32: np.ndarray,
    sample_rate: int,
    response_format: str,
) -> tuple[bytes, str]:
    if response_format == "wav":
        return _wav_bytes(audio_f32, sample_rate), "audio/wav"
    if response_format == "pcm":
        return _pcm_s16le_bytes(audio_f32), "application/octet-stream"
    if response_format in {"mp3", "opus", "aac", "flac"}:
        wav_bytes = _wav_bytes(audio_f32, sample_rate)
        return _ffmpeg_transcode(wav_bytes=wav_bytes, response_format=response_format)
    raise RuntimeError("Unsupported response_format (use wav|mp3|opus|pcm|aac|flac)")


def stream_pcm_audio(
    *,
    engine: MlxTtsEngine,
    infer_lock: threading.Lock,
    text: str,
    voice: str | None,
    speed: float | None,
) -> tuple[Iterable[bytes], int]:
    t0 = time.perf_counter()
    try:

        def inner() -> Generator[bytes, None, None]:
            with infer_lock:
                iterator = iter(engine.iter_chunks(text=text, voice=voice, speed=speed))
                first_chunk = next(iterator, None)
                if first_chunk is None:
                    raise RuntimeError("MLX pipeline produced no audio. Check voice configuration.")
                yield _pcm_s16le_bytes(first_chunk)
                for chunk in iterator:
                    yield _pcm_s16le_bytes(chunk)

        gen = inner()
        first = next(gen)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        def outer() -> Generator[bytes, None, None]:
            yield first
            yield from gen

        return outer(), latency_ms
    except StopIteration:
        raise RuntimeError("MLX pipeline produced no audio. Check voice configuration.") from None


def stream_sse(
    *,
    audio_bytes_iter: Iterable[bytes],
) -> Iterable[bytes]:
    for chunk in audio_bytes_iter:
        if not chunk:
            continue
        payload = {"type": "speech.audio.delta", "audio": base64.b64encode(chunk).decode("ascii")}
        yield _sse_event(payload)
    yield _sse_event(
        {
            "type": "speech.audio.done",
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
    )
