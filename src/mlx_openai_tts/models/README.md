# MLX TTS Model Adapters

This directory contains adapters for different MLX TTS models. Each adapter wraps an MLX model and implements the common interface defined by `ModelAdapter`.

## Adapter Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                        │
│                      (engine.py)                                │
└────────────────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│                    ModelAdapter (base.py)                      │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐    │
│  │             Common Functionality                       │    │
│  │  • Model loading with parameter introspection          │    │
│  │  • Sample rate detection                               │    │
│  │  • Audio generation (full + streaming)                 │    │
│  │  • Parameter capability checking                       │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐    │
│  │             Abstract Methods                           │    │
│  │  • requires_voice                                      │    │
│  │  • resolve_voice()                                     │    │
│  │  • _build_generate_kwargs()                            │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬───────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  KokoroAdapter  │    │ChatterboxAdapter│    │  YourAdapter    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Base Adapter (`base.py`)

**Purpose**: Provides the foundation for all MLX TTS model adapters.

### What It Provides

**Common Functionality** (implemented in `ModelAdapter`):
- **Model Loading**: Loads MLX models via `mlx_audio.tts.utils.load_model()`
- **Parameter Introspection**: Inspects model's `generate()` signature to detect supported parameters
- **Sample Rate Detection**: Extracts sample rate from loaded model
- **Audio Generation**:
  - `synthesize_full()`: Generate complete audio as a single array
  - `iter_chunks()`: Generate audio as an iterable of chunks
- **Capability Checking**: `_supports_generate_param()` checks if model accepts a parameter

**Abstract Methods** (must be implemented by subclasses):
- `requires_voice`: Property indicating if the model needs a voice parameter
- `resolve_voice()`: Resolve voice identifier for the model
- `_build_generate_kwargs()`: Build model-specific kwargs for `generate()`

### Protocols

**`GenerateResult`**:
```python
class GenerateResult(Protocol):
    audio: ArrayLike  # Generated audio chunk
```

**`TtsModel`**:
```python
class TtsModel(Protocol):
    def generate(self, **kwargs: object) -> Iterable[GenerateResult]:
        """Yields GenerateResult objects containing audio chunks."""
```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `spec` | `ModelSpec` | Model specification from registry |
| `model` | `TtsModel | None` | Loaded MLX model instance |
| `sample_rate` | `int` | Audio sample rate in Hz (detected from model) |
| `_generate_params` | `set[str]` | Set of parameter names supported by model.generate() |

---

## Kokoro Adapter (`kokoro.py`)

**Purpose**: Adapter for Kokoro TTS models (neural TTS with voice presets).

### Model Characteristics

**Voice Requirement**: Required (`requires_voice = True`)

Kokoro models always require a voice parameter. Voice must be selected from the configured voices list.

### Voice Resolution

**Behavior**:
1. If `requested_voice` is `None`: Use `spec.default_voice`
2. If `spec.default_voice` is `None`: Raise error (voice required)
3. Validate against `spec.voices` list:
   - If voices list is not empty: Voice must match an entry
   - If voices list is empty: No validation (accepts any string)

**Error Messages**:
- `"voice is required for this model"`: No voice provided and no default
- `"voice must be non-empty"`: Voice string is empty
- `"Unknown voice {voice!r}. Available: {voices}"`: Voice not in allowed list

### Generate Parameters

**Required**:
- `text`: Input text to synthesize

**Conditional**:
- `voice`: Always passed to model (validated beforehand)
- `speed`: Only passed if model supports it

### Example Configuration

```json
{
  "id": "kokoro",
  "repo_id": "mlx-community/Kokoro-82M-bf16",
  "model_type": "kokoro",
  "voices": ["af_bella", "af_nova", "bm_george"],
  "default_voice": "af_bella"
}
```

---

## Chatterbox Adapter (`chatterbox.py`)

**Purpose**: Adapter for Chatterbox TTS models with voice cloning support.

### Model Characteristics

**Voice Requirement**: Optional (`requires_voice = False`)

Chatterbox models can synthesize without a voice parameter. Voice is optional and supports custom cloning from reference audio files.

### Voice Resolution

**Behavior**:

1. **Built-in Voice**: If `requested_voice` is `None` or `"default"` → Returns `None`
   - Uses model's built-in voice
   - No file access required

2. **Voice Cloning**: If `requested_voice` is a string → Resolve to reference audio file
   - Treats as filename in `TTS_VOICE_CLONE_DIR`
   - Supports `.wav` and `.flac` extensions (case-insensitive)
   - If no extension provided: Tries `.wav` first, then `.flac`
   - Filename-only validation (no paths allowed for security)

### Reference Audio Validation

**Path Validation** (prevents directory traversal attacks):
```python
# Rejected values
"."        # Current directory
".."       # Parent directory
"path/to"  # Contains separator
"sub/file" # Contains separator
```

**Extension Validation**:
- Allowed: `.wav`, `.flac` (case-insensitive)
- Error for unsupported extensions

**Directory Checks**:
- `TTS_VOICE_CLONE_DIR` must be set
- Directory must exist and be a valid directory
- File lookup is case-insensitive (cross-platform compatibility)

**File Resolution Flow**:
```
Input: "myvoice"
  ↓
Check extension present?
  ↓ Yes → Validate extension
  ↓ No  → Try "myvoice.wav"
  ↓          → Try "myvoice.flac"
  ↓
Check file exists (case-insensitive)
  ↓
Return absolute path
```

### Generate Parameters

**Required**:
- `text`: Input text to synthesize

**Conditional**:
- `speed`: Only passed if model supports it
- `ref_audio`: If model supports it and voice provided
- `audio_prompt_path`: Alternative parameter name for reference audio
- `audio_prompt`: Alternative parameter name for reference audio

**Parameter Priority** (for reference audio):
1. `ref_audio` (preferred)
2. `audio_prompt_path`
3. `audio_prompt`

### Example Configuration

```json
{
  "id": "chatterbox-turbo",
  "repo_id": "mlx-community/chatterbox-turbo-4bit",
  "model_type": "chatterbox",
  "voices": [],
  "default_voice": null
}
```

**Environment Variables**:
- `TTS_VOICE_CLONE_DIR`: Directory containing reference audio files (e.g., `./voices/`)

### Voice Cloning Example

```
Directory structure:
voices/
  ├── male_voice.wav
  ├── female_voice.flac
  └── CUSTOM.wav

Request examples:
  "voice": "male_voice"        → Uses ./voices/male_voice.wav
  "voice": "female_voice"      → Uses ./voices/female_voice.flac
  "voice": "custom"            → Tries custom.wav, then custom.flac
  "voice": "default"           → Uses built-in voice (no file)
  (omit "voice")              → Uses built-in voice (no file)
```

---

## Adding a New Adapter

### Step 1: Create Adapter File

Create a new file in `src/mlx_openai_tts/models/` (e.g., `my_model.py`):

```python
"""Adapter for [YourModelName] TTS model.

Describe the model's characteristics here (e.g., supports emotion control,
multi-speaker synthesis, etc.).
"""

from __future__ import annotations

from mlx_openai_tts.models.base import ModelAdapter


class MyModelAdapter(ModelAdapter):
    """Adapter for [YourModelName] TTS models.

    Describe the model's behavior here.
    """
    # Optional: Add custom __init__ if you need additional parameters
    # def __init__(self, spec: ModelSpec, *, custom_param: str | None = None):
    #     super().__init__(spec)
    #     self._custom_param = custom_param

    @property
    def requires_voice(self) -> bool:
        """Whether this model requires a voice parameter."""
        return True  # or False depending on your model

    def resolve_voice(self, requested_voice: str | None) -> str | None:
        """Resolve voice identifier for [YourModelName] model.

        Args:
            requested_voice: Voice ID from request, or None.

        Returns:
            Resolved voice identifier (or None if voice is optional).

        Raises:
            RuntimeError: If voice resolution fails.
        """
        if requested_voice is None:
            if self.spec.default_voice is not None:
                return self.spec.default_voice
            if not self.requires_voice:
                return None
            raise RuntimeError("voice is required for this model")

        voice_id = requested_voice.strip()
        if not voice_id:
            raise RuntimeError("voice must be non-empty")

        # Validate against configured voices if list is not empty
        if self.spec.voices and voice_id not in self.spec.voices:
            allowed = ", ".join(sorted(self.spec.voices))
            raise RuntimeError(f"Unknown voice {voice_id!r}. Available: {allowed}")

        return voice_id

    def _build_generate_kwargs(
        self,
        *,
        text: str,
        voice: str | None,
        speed: float | None,
    ) -> dict[str, object]:
        """Build kwargs for [YourModelName] model.generate().

        Args:
            text: Input text to synthesize.
            voice: Resolved voice identifier (or None).
            speed: Playback speed multiplier (or None).

        Returns:
            Dictionary of parameters for model.generate().

        Raises:
            RuntimeError: If model doesn't support required parameters.
        """
        kwargs: dict[str, object] = {"text": text}

        # Add voice if required/supported
        if voice is not None:
            if self._supports_generate_param("voice"):
                kwargs["voice"] = voice
            elif self._supports_generate_param("speaker"):
                kwargs["speaker"] = voice
            else:
                raise RuntimeError("MLX model does not accept voice inputs")

        # Add speed if supported
        if speed is not None and self._supports_generate_param("speed"):
            kwargs["speed"] = float(speed)

        # Add any model-specific parameters
        if self._supports_generate_param("temperature"):
            kwargs["temperature"] = 0.7

        if self._supports_generate_param("emotion"):
            kwargs["emotion"] = "neutral"

        return kwargs
```

### Step 2: Register in Factory

Update `src/mlx_openai_tts/models/__init__.py` to include your adapter:

```python
"""Model adapters for different TTS backends.

Provides factory function to create appropriate adapters based on
model_type (kokoro, chatterbox, your_model).
"""

from __future__ import annotations

from mlx_openai_tts.models.base import ModelAdapter
from mlx_openai_tts.models.chatterbox import ChatterboxAdapter
from mlx_openai_tts.models.kokoro import KokoroAdapter
from mlx_openai_tts.models.my_model import MyModelAdapter  # Add import
from mlx_openai_tts.registry import ModelSpec


def create_adapter(*, spec: ModelSpec, voice_clone_dir: str | None) -> ModelAdapter:
    """Create a model adapter based on specification.

    Args:
        spec: Model specification containing model_type.
        voice_clone_dir: Directory for voice reference audio (chatterbox only).

    Returns:
        Appropriate ModelAdapter subclass instance.

    Raises:
        RuntimeError: If model_type is not supported.
    """
    if spec.model_type == "kokoro":
        return KokoroAdapter(spec)
    if spec.model_type == "chatterbox":
        return ChatterboxAdapter(spec, voice_clone_dir=voice_clone_dir)
    if spec.model_type == "my_model":  # Add your model type
        return MyModelAdapter(spec)
    raise RuntimeError(f"Unsupported model_type {spec.model_type!r}")
```

### Step 3: Update Model Registry

Add your model to `models.json`:

```json
{
  "default_model": "my_model",
  "models": [
    {
      "id": "my_model",
      "repo_id": "mlx-community/YourModelWeights",
      "model_type": "my_model",
      "voices": ["voice1", "voice2", "voice3"],
      "default_voice": "voice1",
      "warmup_text": "Hello from my model."
    }
  ]
}
```

### Step 4: Test Your Adapter

Create unit tests in `tests/test_my_model.py`:

```python
"""Unit tests for MyModel adapter."""

from __future__ import annotations

import unittest

from mlx_openai_tts.models.my_model import MyModelAdapter
from mlx_openai_tts.registry import ModelSpec


class MyModelTests(unittest.TestCase):
    def test_requires_voice(self) -> None:
        """Test that model requires voice parameter."""
        spec = ModelSpec(id="test", repo_id="test", model_type="my_model")
        adapter = MyModelAdapter(spec)
        self.assertTrue(adapter.requires_voice)

    def test_resolve_voice_default(self) -> None:
        """Test resolving to default voice."""
        spec = ModelSpec(
            id="test",
            repo_id="test",
            model_type="my_model",
            voices=["v1", "v2"],
            default_voice="v1",
        )
        adapter = MyModelAdapter(spec)
        self.assertEqual(adapter.resolve_voice(None), "v1")

    def test_resolve_voice_validation(self) -> None:
        """Test voice validation against configured list."""
        spec = ModelSpec(
            id="test",
            repo_id="test",
            model_type="my_model",
            voices=["v1", "v2"],
            default_voice="v1",
        )
        adapter = MyModelAdapter(spec)
        self.assertEqual(adapter.resolve_voice("v2"), "v2")

        with self.assertRaises(RuntimeError) as ctx:
            adapter.resolve_voice("invalid")
        self.assertIn("Unknown voice", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

### Design Decisions Guide

When implementing a new adapter, consider:

**1. Voice Requirement**
- `requires_voice = True`: Model always needs voice (like Kokoro)
- `requires_voice = False`: Voice is optional (like Chatterbox)

**2. Voice Validation**
- **Strict mode** (`voices` list not empty): Validate against list
- **Permissive mode** (`voices` list empty): Accept any string
- **File-based**: Resolve to file paths (like Chatterbox)
- **Preset-based**: Use built-in identifiers (like Kokoro)

**3. Parameter Handling**
- Use `_supports_generate_param()` to check model capabilities
- Map abstract parameters (`text`, `voice`, `speed`) to model-specific names
- Provide helpful error messages when parameters aren't supported

**4. Error Messages**
- Be specific about what went wrong
- Include available options when validation fails
- Use `repr()` for problematic values

**5. Custom Initialization**
- If your adapter needs additional parameters (e.g., `voice_clone_dir`):
  - Add parameters to `__init__`
  - Update factory function to pass them
  - Consider if they should be in `AppConfig` or model-specific

### Common Patterns

**Voice with File Resolution** (like Chatterbox):
```python
def resolve_voice(self, requested_voice: str | None) -> str | None:
    if requested_voice is None or requested_voice == "default":
        return None
    # Resolve to file path in configured directory
    return self._resolve_to_file(requested_voice)
```

**Voice with Preset Validation** (like Kokoro):
```python
def resolve_voice(self, requested_voice: str | None) -> str:
    if requested_voice is None:
        return self.spec.default_voice
    # Validate against configured presets
    if requested_voice not in self.spec.voices:
        raise RuntimeError(f"Unknown voice: {requested_voice}")
    return requested_voice
```

**Multiple Parameter Names** (for compatibility):
```python
def _build_generate_kwargs(self, ..., voice: str | None, ...) -> dict[str, object]:
    kwargs = {"text": text}
    if voice is not None:
        # Try multiple parameter name variations
        for param in ("voice", "speaker", "style"):
            if self._supports_generate_param(param):
                kwargs[param] = voice
                break
    return kwargs
```

## Adapter Comparison

| Feature | `ModelAdapter` (Base) | `KokoroAdapter` | `ChatterboxAdapter` |
|---------|----------------------|------------------|----------------------|
| `requires_voice` | Abstract | Always `True` | Always `False` |
| `resolve_voice()` | Abstract | Preset validation | File resolution + presets |
| `default_voice` | From spec | Required | Optional |
| `voices` list | From spec | Validation enabled | Validation disabled (typically) |
| Custom init | No | No | Yes (`voice_clone_dir`) |
| File access | No | No | Yes (reference audio) |

## Testing

To test all adapters:

```bash
# Run unit tests
uv run python -m unittest tests/test_auth.py tests/test_registry.py

# Test against running server
uv run python tests/test_app.py --base-url http://127.0.0.1:8001
```
