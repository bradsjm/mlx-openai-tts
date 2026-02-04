# MLX OpenAI TTS - Source Code Overview

This directory contains the core implementation of the MLX-based text-to-speech server with an OpenAI-compatible API.

## Package Structure

```
mlx_openai_tts/
├── __init__.py          # Package initialization and version
├── __main__.py          # Command-line entry point
├── auth.py              # Authentication utilities
├── config.py            # Configuration management
├── schemas.py           # API request/response models
├── registry.py          # Model registry and validation
├── engine.py            # Model engine and caching
├── audio.py             # Audio formatting and streaming
├── server.py            # FastAPI server and endpoints
└── models/              # Model adapters
    ├── __init__.py      # Model adapter factory
    ├── base.py          # Base adapter class
    ├── kokoro.py        # Kokoro TTS adapter
    └── chatterbox.py    # Chatterbox TTS adapter
```

## File Descriptions

### `__init__.py`

**Purpose**: Package initialization and version definition.

**Key Exports**:
- `__version__`: Package version string

**Usage**:
```python
from mlx_openai_tts import __version__
print(__version__)  # "0.3.0"
```

### `__main__.py`

**Purpose**: Command-line entry point for running the server as a Python module.

**Key Functions**:
- `main()`: Entry point that configures and starts uvicorn

**Usage**:
```bash
python -m mlx_openai_tts
```

---

### `auth.py`

**Purpose**: Authentication utilities for API access control.

**Key Functions**:
- `require_auth(authorization, api_key)`: Validate Bearer token using constant-time comparison

**Features**:
- Uses `hmac.compare_digest()` to prevent timing attacks
- Returns early if API key is not configured (auth disabled)
- Raises `HTTPException` with 401 status for missing or invalid tokens

---

### `config.py`

**Purpose**: Load and validate configuration from environment variables.

**Key Functions**:
- `load_config()`: Load all configuration options from environment
- `_env_int()`: Parse integer environment variable with validation
- `_env_str()`: Parse string environment variable with default
- `_env_float()`: Parse float environment variable with validation

**Data Classes**:
- `AppConfig`: Frozen dataclass containing all configuration options

**Environment Variables Handled**:
- `HOST`, `PORT`: Server binding
- `API_KEY`: Authentication token
- `TTS_MODELS_JSON`: Path to models configuration file
- `TTS_MAX_CHARS`: Maximum input text length
- `TTS_WARMUP_TEXT`: Startup warmup text
- `TTS_MLX_STRICT`: Strict mode flag for model loading
- `TTS_VOICE_CLONE_DIR`: Voice cloning directory
- `FFMPEG_BIN`: Path to ffmpeg binary
- `FFMPEG_TIMEOUT_SECONDS`: Transcode timeout

---

### `schemas.py`

**Purpose**: Pydantic request/response models for the OpenAI-compatible API.

**Key Models**:
- `AudioSpeechRequest`: Request model for `/v1/audio/speech` endpoint
- `VoicePayload`: Flexible voice identifier object
- `ModelListItem`: Individual model entry
- `ModelListResponse`: Response for `/v1/models` endpoint
- `HealthResponse`: Response for health check endpoints

**Type Aliases**:
- `ResponseFormat`: Literal for supported audio formats
- `StreamFormat`: Literal for streaming modes

**Validation**:
- Input text minimum length: 1
- Speed range: 0.25 to 4.0
- Response format defaults: mp3, audio

---

### `registry.py`

**Purpose**: Load and validate model specifications from `models.json`.

**Key Functions**:
- `load_registry(path)`: Load and validate model registry from file
- `load_registry_payload(payload)`: Validate parsed JSON structure

**Data Classes**:
- `ModelSpec`: Single model specification
- `ModelRegistry`: Container for all model specifications
- `ResolvedRegistry`: Validated registry with indexed lookups

**Validation**:
- At least one model required
- No duplicate model IDs
- Non-empty model IDs and repository IDs
- Default voice must be in voices list (if specified)
- Default model must exist in models list

---

### `engine.py`

**Purpose**: Model loading, caching, and thread-safe inference management.

**Key Classes**:
- `MlxTtsEngine`: Wrapper around `ModelAdapter` for audio generation
- `ModelManager`: Manages multiple models with caching and switching

**ModelManager Features**:
- Preloads all models on startup
- Caches loaded engines by model ID
- Uses double-checked locking for thread-safe model switching
- Serializes inference with a lock for predictable latency
- Supports warmup synthesis for default model

**Key Methods** (ModelManager):
- `load_all()`: Preload all models in registry
- `load_default()`: Load and warm up the default model
- `get_engine()`: Get or load engine for a specific model (thread-safe)
- `resolve_voice()`: Resolve voice identifier for a model

---

### `audio.py`

**Purpose**: Audio format conversion, encoding, and streaming utilities.

**Key Functions**:
- `build_full_response()`: Generate complete audio response in specified format
- `stream_pcm_audio()`: Stream audio as PCM16 chunks with latency measurement
- `stream_sse()`: Convert audio stream to Server-Sent Events

**Format Conversion**:
- `coerce_audio_1d_float32()`: Normalize audio to 1-D float32 numpy array
- `_wav_bytes()`: Convert float32 to WAV bytes
- `_pcm_s16le_bytes()`: Convert float32 to PCM16 signed little-endian
- `_ffmpeg_transcode()`: Transcode WAV to mp3/opus/aac/flac using ffmpeg

**Streaming**:
- Nested generator pattern for measuring time-to-first-byte
- SSE event formatting with base64 encoding
- Headers for latency reporting

**Dependencies**:
- `numpy`: Array operations
- `soundfile`: WAV encoding
- `ffmpeg`: Required for compressed formats (mp3, opus, aac, flac)

---

### `server.py`

**Purpose**: FastAPI server with OpenAI-compatible endpoints.

**Key Functions**:
- `main()`: Configure and start uvicorn server

**Endpoints**:
- `GET /`: Health check
- `GET /health`: Health check
- `GET /v1`: Health check
- `GET /v1/models`: List available models
- `POST /v1/audio/speech`: Generate audio from text

**Helper Functions**:
- `_normalize_text()`: Normalize whitespace in input text
- `_parse_voice_id()`: Parse voice from string or VoicePayload
- `_configure_loguru()`: Route loguru logs to uvicorn
- `_require_auth()`: Validate authorization if API key is configured

**Lifespan Management**:
- `lifespan()`: Async context manager for app startup/shutdown
  - Loads configuration
  - Creates model manager
  - Preloads all models
  - Warms up default model

**Response Handling**:
- Supports both streaming and non-streaming modes
- Multiple audio formats via format conversion
- Custom headers: `X-Model`, `X-Voice`, `X-Sample-Rate`, `X-Latency-Ms`
- Error handling with HTTP exceptions

---

### `models/__init__.py`

**Purpose**: Model adapter factory and subpackage initialization.

**Key Functions**:
- `create_adapter(spec, voice_clone_dir)`: Factory to create appropriate adapter

**Factory Logic**:
- Returns `KokoroAdapter` for `model_type: "kokoro"`
- Returns `ChatterboxAdapter` for `model_type: "chatterbox"`
- Raises error for unsupported model types

---

### `models/base.py`

**Purpose**: Abstract base class and protocol definitions for model adapters.

**Key Protocols**:
- `GenerateResult`: Protocol for MLX model generation result
- `TtsModel`: Protocol for MLX TTS model interface

**Key Classes**:
- `ModelAdapter`: Base class for model adapters

**ModelAdapter Features**:
- Model loading with parameter introspection
- Sample rate detection from loaded model
- Audio generation (full and chunked streaming)
- Voice resolution (abstract, implemented by subclasses)

**Key Methods**:
- `load(strict)`: Load MLX model and inspect generate parameters
- `iter_chunks(text, voice, speed)`: Generate streaming audio chunks
- `synthesize_full(text, voice, speed)`: Generate complete audio
- `_supports_generate_param(name)`: Check if model supports a parameter
- `_build_generate_kwargs(text, voice, speed)`: Build kwargs (abstract)

**Abstract Methods** (must be implemented by subclasses):
- `requires_voice`: Whether model requires a voice parameter
- `resolve_voice(requested_voice)`: Resolve voice identifier
- `_build_generate_kwargs(...)`: Build model-specific parameters

---

### `models/kokoro.py`

**Purpose**: Adapter for Kokoro TTS models.

**Key Classes**:
- `KokoroAdapter(ModelAdapter)`: Kokoro-specific model adapter

**Behavior**:
- Requires voice parameter (`requires_voice = True`)
- Validates voice against configured voices list
- Uses default voice from model spec if not provided

**Key Methods**:
- `resolve_voice(requested_voice)`:
  - Returns `spec.default_voice` if not provided
  - Validates against `spec.voices` list
  - Raises error for invalid voice

- `_build_generate_kwargs(text, voice, speed)`:
  - Requires voice parameter
  - Adds `voice` if model supports it
  - Adds `speed` if model supports it

---

### `models/chatterbox.py`

**Purpose**: Adapter for Chatterbox TTS models with voice cloning support.

**Key Classes**:
- `ChatterboxAdapter(ModelAdapter)`: Chatterbox-specific model adapter

**Behavior**:
- Does not require voice parameter (`requires_voice = False`)
- Supports voice cloning from reference audio files
- Uses built-in voice if not provided or set to "default"

**Voice Cloning**:
- `voice` parameter treated as filename in `TTS_VOICE_CLONE_DIR`
- Supported extensions: `.wav` and `.flac` (case-insensitive)
- Filename-only validation (no paths allowed)
- Case-insensitive file lookup

**Key Methods**:
- `resolve_voice(requested_voice)`:
  - Returns `None` for "default" or `None` (use built-in voice)
  - Resolves to reference audio path for custom voices
  - Validates file extension and existence

- `_resolve_ref_audio(voice)`:
  - Validates voice is a filename (not a path)
  - Checks `TTS_VOICE_CLONE_DIR` exists
  - Finds file by name (case-insensitive)
  - Tries `.wav`, then `.flac` if no extension provided

- `_find_voice_file(directory, filename, raise_if_missing)`:
  - Direct path lookup
  - Case-insensitive directory scan for fallback

- `_build_generate_kwargs(text, voice, speed)`:
  - Adds `speed` if model supports it
  - Adds reference audio as `ref_audio`, `audio_prompt_path`, or `audio_prompt`
  - Supports multiple parameter name variations

---

## Data Flow

### Startup Flow

```
Load Config
  ↓
Load Registry (models.json)
  ↓
Create Model Manager
  ↓
Preload All Models
  ↓
Warm Up Default Model
  ↓
Start FastAPI Server
```

### Request Flow

```
HTTP Request
  ↓
Validate Auth (if API_KEY set)
  ↓
Parse Request (schemas)
  ↓
Get Engine (ModelManager, thread-safe)
  ↓
Resolve Voice (Adapter)
  ↓
Synthesize Audio (MLX)
  ↓
Format/Stream Audio (audio.py)
  ↓
HTTP Response with Headers
```

### Streaming Flow

```
HTTP Request
  ↓
Get/Lock Engine
  ↓
Start Generation Under Lock
  ↓
Yield First Chunk → Measure Latency
  ↓
Yield Remaining Chunks (still under lock)
  ↓
Return to Unlock
  ↓
Stream Remaining Chunks
```
