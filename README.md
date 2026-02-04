# MLX OpenAI-Compatible TTS Server

A local text-to-speech server that runs MLX models on Apple Silicon, exposing an OpenAI-compatible API.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Application                            │
│                         (curl, httpx, etc.)                             │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │ HTTP/JSON
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Server                                     │
│                    (single uvicorn worker)                              │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              Authentication Layer                                 │  │
│  │        (Bearer token, optional)                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              API Endpoints                                        │  │
│  │  • GET  /v1/models (list available models)                        │  │
│  │  • GET  /health, /v1 (health check)                               │  │
│  │  • POST /v1/audio/speech (generate audio)                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              Model Manager                                        │  │
│  │  • Model cache (preloads all models on startup)                   │  │
│  │  • Thread-safe model switching (serialized inference)             │  │
│  └───────────────┬───────────────────────────────────────────────────┘  │
│                  │                                                      │
│  ┌───────────────┴───────────────────────────────────────────────────┐  │
│  │              MLX Audio Engines                                    │  │
│  │  ┌─────────────┐  ┌─────────────────────┐                         │  │
│  │  │  Kokoro     │  │  Chatterbox         │                         │  │
│  │  │  Adapter    │  │  Adapter            │                         │  │
│  │  └─────────────┘  └─────────────────────┘                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Requirements

- macOS Apple Silicon (arm64)
- Python 3.10–3.12
- Homebrew (optional, for compressed audio formats):
  - `brew install ffmpeg` (required for mp3, opus, aac, flac)
  - `brew install libsndfile` (sometimes needed for soundfile)

## Quick Start

```bash
cd mlx-openai-tts

# Create and activate virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install with dev dependencies
uv pip install -e .[dev]

# Start the server
uv run mlx-openai-tts
```

The server will:
1. Load all models from `models.json`
2. Warm up the default model
3. Listen on `http://0.0.0.0:8001` (default)

## Configuration

Configure via environment variables:

```bash
# Server binding
export HOST=0.0.0.0
export PORT=8001

# Authentication (optional; if unset, auth is disabled)
export API_KEY=local-dev

# Model registry
export TTS_MODELS_JSON=./models.json
export TTS_MAX_CHARS=4096
export TTS_WARMUP_TEXT="Hello from MLX TTS."
export TTS_MLX_STRICT=false

# Voice cloning directory (for Chatterbox models)
export TTS_VOICE_CLONE_DIR=./voices
```

### Environment Variables

| Variable | Default | Description |
|----------|----------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8001` | Server port |
| `API_KEY` | `None` | Bearer token for authentication (disabled if unset) |
| `TTS_MODELS_JSON` | `models.json` | Path to model configuration file |
| `TTS_MAX_CHARS` | `4096` | Maximum input text length in characters |
| `TTS_WARMUP_TEXT` | `Hello from MLX TTS.` | Warmup text for default model on startup |
| `TTS_MLX_STRICT` | `false` | Strict mode for MLX model loading (`1`, `true`, `yes`, `on` to enable) |
| `TTS_VOICE_CLONE_DIR` | `None` | Directory containing reference audio files for voice cloning |
| `FFMPEG_BIN` | `auto-detect` | Path to ffmpeg binary |
| `FFMPEG_TIMEOUT_SECONDS` | `30.0` | Transcode timeout in seconds |

## Model Configuration

Edit `models.json` to define available models:

```json
{
  "default_model": "kokoro",
  "models": [
    {
      "id": "kokoro",
      "repo_id": "mlx-community/Kokoro-82M-bf16",
      "model_type": "kokoro",
      "voices": ["af_bella", "af_nova", "bm_george"],
      "default_voice": "af_bella",
      "warmup_text": "Hello from Kokoro."
    },
    {
      "id": "chatterbox-turbo",
      "repo_id": "mlx-community/chatterbox-turbo-4bit",
      "model_type": "chatterbox",
      "voices": [],
      "default_voice": null,
      "warmup_text": "Hello from Chatterbox."
    }
  ]
}
```

### Model Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | OpenAI `model` name exposed to clients |
| `repo_id` | string | Yes | Hugging Face repository ID for MLX weights |
| `model_type` | string | No | Adapter type: `kokoro` or `chatterbox` (default: `kokoro`) |
| `voices` | array | No | List of allowed voice IDs. Empty array disables validation |
| `default_voice` | string \| null | No | Default voice for this model |
| `warmup_text` | string \| null | No | Warmup text used only for default model on startup |

### Model Behavior

**Kokoro Models** (`model_type: "kokoro"`):
- Require a `voice` parameter
- Voice must match an entry in the `voices` list
- Use built-in preset voices from the model

**Chatterbox Models** (`model_type: "chatterbox"`):
- Voice parameter is optional
- If `voice` is omitted or set to `"default"`, uses built-in voice
- If `voice` is provided, treated as a filename in `TTS_VOICE_CLONE_DIR`
- Supported extensions: `.wav` and `.flac` (case-insensitive)
- If no extension is provided, tries `.wav` first, then `.flac`
- Voice must be a filename only (no paths allowed)

## API Usage

### Request Format

```bash
POST /v1/audio/speech
Content-Type: application/json
Authorization: Bearer <api_key>  # optional

{
  "model": "kokoro",           # uses default_model if omitted
  "voice": "af_bella",        # required for Kokoro models
  "input": "Hello world!",     # required, max chars per TTS_MAX_CHARS
  "response_format": "wav",     # wav, mp3, opus, aac, flac, pcm
  "stream_format": "audio",     # audio (streaming) or sse (events)
  "speed": 1.0                # 0.25 - 4.0
}
```

### Supported Formats

| `response_format` | Description | External Dependencies |
|------------------|-------------|----------------------|
| `wav` | Uncompressed WAV | None |
| `pcm` | Raw PCM16 signed little-endian | None |
| `mp3` | MP3 encoded | ffmpeg |
| `opus` | Opus encoded in OGG container | ffmpeg |
| `aac` | AAC encoded in ADTS container | ffmpeg |
| `flac` | FLAC encoded | ffmpeg |

### Streaming Modes

**Audio Streaming** (`stream_format: "audio"`):
- Streams audio chunks as they are generated
- Lowest latency for playback
- Recommended for real-time applications

```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","voice":"af_bella","input":"Hello!","response_format":"pcm","stream_format":"audio"}' \
  --output out.pcm
```

**SSE Streaming** (`stream_format: "sse"`):
- Wraps audio chunks in Server-Sent Events
- Base64-encoded audio in each event
- Ends with `speech.audio.done` event

```
data: {"type":"speech.audio.delta","audio":"<base64_audio>"}

data: {"type":"speech.audio.done","usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}
```

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Model` | Model ID used for synthesis |
| `X-Voice` | Voice ID used for synthesis |
| `X-Sample-Rate` | Audio sample rate in Hz |
| `X-Latency-Ms` | Time to first audio in milliseconds |

### Endpoints

**List Models**
```bash
GET /v1/models
Authorization: Bearer <api_key>  # optional

Response:
{
  "object": "list",
  "data": [
    {"id": "kokoro", "object": "model", "owned_by": "local", "permission": []},
    {"id": "chatterbox-turbo", "object": "model", "owned_by": "local", "permission": []}
  ],
  "default_model": "kokoro",
  "default_voice": "af_bella"
}
```

**Health Check**
```bash
GET /health
Authorization: Bearer <api_key>  # optional

Response:
{
  "status": "ok",
  "name": "mlx-openai-tts",
  "version": "0.3.0",
  "active_model": "kokoro",
  "repo_id": "mlx-community/Kokoro-82M-bf16",
  "default_voice": "af_bella",
  "sample_rate": 24000
}
```

## Running Without Virtual Environment

```bash
# Using uvx (no venv required)
uvx --from . mlx-openai-tts

# Using run script
./scripts/run.sh
```

## Testing

### OpenAI-Compatible API Tests

```bash
uv run python tests/test_app.py \
  --base-url http://127.0.0.1:8001 \
  --api-key local-dev \
  --voice af_bella
```

This tests:
- Health checks
- Model listing
- Audio synthesis in multiple formats
- Streaming (both audio and SSE)
- Error handling

### Chatterbox Voice Cloning Tests

```bash
uv run python tests/test_chatterbox.py \
  --base-url http://127.0.0.1:8001 \
  --api-key local-dev
```

This tests:
- Chatterbox default voice
- Voice cloning from reference audio
- Various output formats

### Unit Tests

```bash
uv run python -m unittest tests/test_auth.py tests/test_registry.py
```

This tests:
- Authentication logic
- Model registry validation
- Voice resolution

## Auto-Start on Login (LaunchAgent)

Install as a macOS LaunchAgent to auto-start the server on login:

```bash
# Install and start immediately
./scripts/install-launchagent.sh

# Uninstall
./scripts/uninstall-launchagent.sh
```

The LaunchAgent:
- Sets `PATH` to include `/opt/homebrew/bin` for ffmpeg availability
- Runs under the current user's LaunchAgents
- Persists across system restarts

## Development

### Code Formatting

```bash
uv run ruff format .
```

### Linting

```bash
uv run ruff check .
```

### Type Checking

```bash
uv run pyright
```

## Performance Notes

- **Single Worker**: Server runs with `--workers 1` for predictable inference latency on Apple Silicon
- **Serialized Inference**: Thread lock ensures only one inference runs at a time
- **Preloaded Models**: All models are loaded on startup to reduce first-request latency
- **Memory Usage**: Each loaded model remains in memory; consider this when adding many models
- **Startup Time**: Startup time scales with number of models in `models.json`

## Troubleshooting

**MLX weight shape errors on startup**:
- Models must match the installed `mlx-audio` version
- Try switching to a different model in `models.json`
- Set `TTS_MLX_STRICT=false` to allow relaxed loading

**"ffmpeg not found" error**:
- Install via `brew install ffmpeg`
- Or set `FFMPEG_BIN` to the full path to your ffmpeg binary

**Voice validation errors**:
- Check that voice IDs match the `voices` list in `models.json`
- For Chatterbox, ensure `TTS_VOICE_CLONE_DIR` exists and contains valid audio files

**Slow first request**:
- Models are preloaded on startup, but the default model is warmed with synthesis
- Adjust `TTS_WARMUP_TEXT` to warm up with your typical use case

## License

MIT License - see LICENSE file for details.
