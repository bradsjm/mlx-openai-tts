# MLX OpenAI-Compatible TTS Server (local dev)

Runs MLX TTS models in-process (single uvicorn worker), exposing an OpenAI-compatible endpoint:

- `POST /v1/audio/speech` (JSON in, audio bytes out; optional streaming)
- `GET /v1/models`

Designed for local development with limited requests; optimized for Apple Silicon performance via MLX.

## Prereqs

- macOS Apple Silicon (arm64)
- Python 3.10–3.12
- Homebrew (optional): `brew install libsndfile` (sometimes needed for `soundfile`)

## Setup

```bash
cd mlx-openai-tts
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .[dev]
```

## Configure

```bash
export HOST=0.0.0.0
export PORT=8001

# Optional auth. If unset, auth is disabled.
export API_KEY=local-dev

# Model registry
export TTS_MODELS_JSON=./models.json
export TTS_MAX_CHARS=4096
export TTS_WARMUP_TEXT="Hello from MLX TTS."
export TTS_MLX_STRICT=false
export TTS_VOICE_CLONE_DIR=./voices
```

## Models

Edit `models.json` to curate which MLX models are available. Each entry supports:

- `id`: OpenAI `model` name exposed to clients.
- `repo_id`: Hugging Face repo ID for MLX weights.
- `model_type`: Which adapter to use (`kokoro` or `chatterbox`).
- `voices`: Array of allowed voice IDs. Empty array disables voice validation.
- `default_voice`: Optional default voice (required if you want to omit `voice` in requests).
- `warmup_text`: Optional per-model warmup text (only used for the default model on startup).

## Run

```bash
uv run mlx-openai-tts
```

## uvx (no venv)

```bash
uvx --from . mlx-openai-tts
```

## Test the API against spec

Run the test client (requires `httpx` from `requirements.txt`):

```bash
uv run python tests/test_app.py --base-url http://127.0.0.1:8001 --api-key local-dev --voice af_bella
```

## Unit tests

```bash
uv run python -m unittest tests/test_auth.py tests/test_registry.py
```

## Chatterbox integration test

Run the Chatterbox default voice checks (requires a running server with the
`chatterbox-turbo` model configured in `models.json`):

```bash
uv run python tests/test_chatterbox.py --base-url http://127.0.0.1:8001 --api-key local-dev
```

## Auto-start on login (LaunchAgent)

```bash
./scripts/install-launchagent.sh
```

This installs `~/Library/LaunchAgents/com.local.mlx-openai-tts.plist` and starts it immediately. To remove:

```bash
./scripts/uninstall-launchagent.sh
```

## Example (curl)

```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H "Authorization: Bearer local-dev" \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","voice":"af_bella","input":"Hello!","response_format":"wav","speed":1.0}' \
  --output out.wav
```

## Streaming (recommended for lowest latency)

Use `response_format=pcm` with `stream_format=audio` for true streaming:

```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H "Authorization: Bearer local-dev" \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","voice":"af_bella","input":"Hello!","response_format":"pcm","stream_format":"audio"}' \
  --output out.pcm
```

If you prefer SSE-style streaming, use `stream_format=sse` (base64 audio chunks).

## Notes

- Supported `response_format`: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`.
- The `model` must be one of the IDs listed in `models.json` (see `/v1/models`).
- If `voices` is empty for a model, the server accepts any voice string without validation. Otherwise voices must match the list.
- If a model has no `default_voice`, the request must include `voice` (unless the model adapter allows missing voices).
- If you see MLX weight shape errors on startup, switch to a different model in `models.json` and/or set `TTS_MLX_STRICT=false`.
- Models are preloaded on startup to reduce first-request latency; expect higher startup time and memory usage.
- This server runs with `--workers 1` and serializes inference with a lock for predictable latency on Apple Silicon.
- For `response_format=mp3`, `opus`, `aac`, or `flac`, `ffmpeg` is required (`brew install ffmpeg`).
- The LaunchAgent template sets `PATH` to include `/opt/homebrew/bin` so `ffmpeg` is available under launchd.

### Chatterbox voice cloning

If you register a `chatterbox` model, the `voice` parameter is treated as a filename in
`TTS_VOICE_CLONE_DIR`.

- If `voice` is omitted or set to `default`, Chatterbox uses its built-in voice.
- Allowed extensions: `.wav` and `.flac` (case-insensitive).
- If no extension is provided, `.wav` is tried first, then `.flac`.
- The `voice` value must be a filename only (no paths).

## Formatting

```bash
uv run ruff format .
```

## Linting

```bash
uv run ruff check .
```

## Type checking

```bash
uv run pyright
```
