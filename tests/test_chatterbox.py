#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import httpx


@dataclass
class Settings:
    base_url: str
    api_key: str | None
    timeout: float
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
    return client.post(path, json=payload)


def run_checks(settings: Settings) -> None:
    with httpx.Client(
        base_url=settings.base_url, headers=_headers(settings), timeout=settings.timeout
    ) as client:
        health = _get_json(client, "/health")
        _assert(health.get("status") == "ok", "health.status != ok")

        models = _get_json(client, "/v1/models")
        _assert(models.get("object") == "list", "models.object != list")
        data = models.get("data") or []
        _assert(any(item.get("id") == settings.model for item in data), "model id not listed")

        t0 = time.perf_counter()
        wav = _post_json(
            client,
            "/v1/audio/speech",
            {
                "model": settings.model,
                "input": "Hello from chatterbox default voice.",
                "response_format": "wav",
            },
        )
        wav.raise_for_status()
        _assert(
            wav.headers.get("content-type", "").startswith("audio/"),
            "wav content-type not audio/*",
        )
        _assert(len(wav.content) > 44, "wav content too small")
        elapsed = time.perf_counter() - t0
        print(f"wav_default_voice: {elapsed:.4f} s")

        t0 = time.perf_counter()
        wav_default = _post_json(
            client,
            "/v1/audio/speech",
            {
                "model": settings.model,
                "voice": "default",
                "input": "Hello from chatterbox explicit default voice.",
                "response_format": "wav",
            },
        )
        wav_default.raise_for_status()
        _assert(
            wav_default.headers.get("content-type", "").startswith("audio/"),
            "wav content-type not audio/*",
        )
        _assert(len(wav_default.content) > 44, "wav content too small")
        elapsed = time.perf_counter() - t0
        print(f"wav_explicit_default_voice: {elapsed:.4f} s")

        stream = client.post(
            "/v1/audio/speech",
            json={
                "model": settings.model,
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
        total_bytes = 0
        for chunk in stream.iter_bytes():
            total_bytes += len(chunk)
        _assert(total_bytes > 0, "pcm stream produced no bytes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test chatterbox default voice")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="Server base URL")
    parser.add_argument("--api-key", default=None, help="API key (if enabled)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout seconds")
    parser.add_argument(
        "--model", default="chatterbox-turbo", help="Model name (must be listed in /v1/models)"
    )
    args = parser.parse_args()

    settings = Settings(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        model=args.model,
    )
    try:
        run_checks(settings)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: chatterbox default voice checks succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
