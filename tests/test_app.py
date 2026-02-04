#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass

import httpx


@dataclass
class Settings:
    base_url: str
    api_key: str | None
    timeout: float
    voice: str
    model: str


def _headers(settings: Settings) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    return headers


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _get_json(client: httpx.Client, path: str) -> dict:
    resp = client.get(path)
    resp.raise_for_status()
    return resp.json()


def _post_json(client: httpx.Client, path: str, payload: dict) -> httpx.Response:
    resp = client.post(path, json=payload)
    return resp


def _iter_sse_lines(response: httpx.Response) -> Iterable[str]:
    for line in response.iter_lines():
        if line is None:
            continue
        line = line.strip()
        if not line:
            continue
        yield line


def run_checks(settings: Settings) -> None:
    with httpx.Client(
        base_url=settings.base_url, headers=_headers(settings), timeout=settings.timeout
    ) as client:
        timings: list[tuple[str, float]] = []
        long_text = (
            "In a quiet studio, the narrator reads a full sentence to measure speech speed, "
            "clarity, and timing across different model settings."
        )
        long_words = len(long_text.split())
        health = _get_json(client, "/health")
        _assert(health.get("status") == "ok", "health.status != ok")

        models = _get_json(client, "/v1/models")
        _assert(models.get("object") == "list", "models.object != list")
        data = models.get("data") or []
        _assert(any(item.get("id") == settings.model for item in data), "model id not listed")

        # Model validation
        bad_model = _post_json(
            client,
            "/v1/audio/speech",
            {
                "model": "not-mlx",
                "voice": settings.voice,
                "input": "test",
            },
        )
        _assert(bad_model.status_code == 400, "invalid model should return 400")

        # Voice validation
        bad_voice = _post_json(
            client,
            "/v1/audio/speech",
            {
                "model": settings.model,
                "voice": "bad_voice",
                "input": "test",
            },
        )
        _assert(bad_voice.status_code == 400, "invalid voice should return 400")

        # Non-streaming audio (wav)
        t0 = time.perf_counter()
        wav = _post_json(
            client,
            "/v1/audio/speech",
            {
                "model": settings.model,
                "voice": settings.voice,
                "input": "Hello from mlx test.",
                "response_format": "wav",
            },
        )
        wav.raise_for_status()
        timings.append(("wav_non_stream", time.perf_counter() - t0))
        _assert(
            wav.headers.get("content-type", "").startswith("audio/"), "wav content-type not audio/*"
        )
        _assert(len(wav.content) > 44, "wav content too small")

        # Long-form sentence benchmark (wav)
        t0 = time.perf_counter()
        long_wav = _post_json(
            client,
            "/v1/audio/speech",
            {
                "model": settings.model,
                "voice": settings.voice,
                "input": long_text,
                "response_format": "wav",
            },
        )
        long_wav.raise_for_status()
        elapsed = time.perf_counter() - t0
        timings.append(("long_wav_non_stream", elapsed))
        timings.append(("long_wav_words_per_sec", long_words / max(elapsed, 1e-6)))

        # Non-streaming audio (compressed formats)
        format_expectations = {
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "aac": "audio/aac",
            "flac": "audio/flac",
        }
        for fmt, expected_ct in format_expectations.items():
            t0 = time.perf_counter()
            resp = _post_json(
                client,
                "/v1/audio/speech",
                {
                    "model": settings.model,
                    "voice": settings.voice,
                    "input": f"Format test {fmt}.",
                    "response_format": fmt,
                },
            )
            resp.raise_for_status()
            timings.append((f"{fmt}_non_stream", time.perf_counter() - t0))
            _assert(
                resp.headers.get("content-type", "") == expected_ct,
                f"{fmt} content-type mismatch (expected {expected_ct})",
            )
            _assert(len(resp.content) > 0, f"{fmt} content empty")

        # Optional alternate model (if provided by registry)
        alt_model = None
        for item in data:
            model_id = item.get("id")
            if model_id and model_id != settings.model:
                alt_model = model_id
                break
        if alt_model:
            t0 = time.perf_counter()
            resp = _post_json(
                client,
                "/v1/audio/speech",
                {
                    "model": alt_model,
                    "voice": settings.voice,
                    "input": f"Alternate model test {alt_model}.",
                    "response_format": "wav",
                },
            )
            if resp.status_code < 400:
                timings.append((f"alt_model_{alt_model}_wav", time.perf_counter() - t0))
                _assert(
                    resp.headers.get("content-type", "").startswith("audio/"),
                    "alt model response content-type not audio/*",
                )

        # Streaming audio (pcm)
        t0 = time.perf_counter()
        stream = client.post(
            "/v1/audio/speech",
            json={
                "model": settings.model,
                "voice": settings.voice,
                "input": "Streaming test audio.",
                "response_format": "pcm",
                "stream_format": "audio",
            },
        )
        stream.raise_for_status()
        _assert(
            stream.headers.get("content-type", "") == "application/octet-stream",
            "pcm stream content-type mismatch",
        )
        first_chunk_time = None
        total_bytes = 0
        for chunk in stream.iter_bytes():
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            total_bytes += len(chunk)
        timings.append(("pcm_stream_ttfb", (first_chunk_time or time.perf_counter()) - t0))
        timings.append(("pcm_stream_total", time.perf_counter() - t0))
        _assert(total_bytes > 0, "pcm stream produced no bytes")

        # SSE streaming
        t0 = time.perf_counter()
        sse = client.post(
            "/v1/audio/speech",
            json={
                "model": settings.model,
                "voice": settings.voice,
                "input": "SSE streaming test.",
                "response_format": "pcm",
                "stream_format": "sse",
            },
        )
        sse.raise_for_status()
        _assert(
            sse.headers.get("content-type", "").startswith("text/event-stream"),
            "sse content-type mismatch",
        )

        saw_delta = False
        saw_done = False
        decoded_bytes = 0
        for line in _iter_sse_lines(sse):
            if not line.startswith("data:"):
                continue
            payload = json.loads(line[len("data:") :].strip())
            event_type = payload.get("type")
            if event_type == "speech.audio.delta":
                audio_b64 = payload.get("audio")
                _assert(
                    isinstance(audio_b64, str) and bool(audio_b64),
                    "delta audio missing",
                )
                decoded_bytes += len(base64.b64decode(audio_b64))
                saw_delta = True
            elif event_type == "speech.audio.done":
                usage = payload.get("usage", {})
                _assert(
                    {"input_tokens", "output_tokens", "total_tokens"} <= set(usage.keys()),
                    "done usage missing fields",
                )
                saw_done = True
                break

        _assert(saw_delta, "no SSE delta events observed")
        _assert(saw_done, "no SSE done event observed")
        _assert(decoded_bytes > 0, "SSE audio bytes decoded to zero")
        timings.append(("sse_stream_total", time.perf_counter() - t0))

        if timings:
            print("BENCHMARKS:")
            for label, value in timings:
                if "words_per_sec" in label:
                    print(f"{label}: {value:.2f} wps")
                else:
                    print(f"{label}: {value:.4f} s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test OpenAI-compatible MLX TTS server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="Server base URL")
    parser.add_argument("--api-key", default=None, help="API key (if enabled)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout seconds")
    parser.add_argument("--voice", default="af_bella", help="Voice ID (with prefix)")
    parser.add_argument(
        "--model", default="kokoro", help="Model name (must be listed in /v1/models)"
    )
    args = parser.parse_args()

    settings = Settings(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        voice=args.voice,
        model=args.model,
    )
    try:
        run_checks(settings)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: all checks succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
