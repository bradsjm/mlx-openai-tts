"""Entry point for MLX TTS server.

Allows running `python -m mlx_openai_tts`.
"""

from __future__ import annotations

from .server import main

if __name__ == "__main__":
    raise SystemExit(main())
