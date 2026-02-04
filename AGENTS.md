# Repository Guidelines for Agents

This file is for automated coding agents working in this repo. Keep changes minimal,
follow the existing code style, and prefer uv-based commands.

## Project Layout
- `src/mlx_openai_tts/`: Core server, runtime, and model logic. Entry point is
  `mlx_openai_tts.server:main`.
- `tests/`: Unit tests (unittest) and the API test client (`tests/test_app.py`).
- `scripts/`: Helper scripts (`install-launchagent.sh`, `uninstall-launchagent.sh`, `run.sh`).
- `launchd/`: LaunchAgent templates and plist files.
- `models.json`: Local model registry consumed by the server at runtime.
- `README.md`: Setup, configuration, API usage, and developer commands.

## Cursor/Copilot Rules
- No Cursor rules found in `.cursor/rules/` or `.cursorrules`.
- No Copilot instructions found in `.github/copilot-instructions.md`.

## Environment and Setup
- Recommended Python: 3.12 (supports 3.10+ per `pyproject.toml`).
- Create a dev venv and install dependencies:
  - `uv venv --python 3.12`
  - `uv pip install -e .[dev]`
- Run the server (venv): `uv run mlx-openai-tts`
- Run the server (no venv): `uvx --from . mlx-openai-tts`
- Alternate run script (uses uvicorn directly if available): `./scripts/run.sh`

## Build, Lint, Format, Type Check
- Format: `uv run ruff format .`
- Lint: `uv run ruff check .`
- Type check: `uv run pyright`
- There is no custom build step; the project is a standard setuptools package.

## Testing Commands
Unit tests use `unittest`.
- Run all unit tests:
  - `uv run python -m unittest tests/test_auth.py tests/test_registry.py`
- Run a single test file:
  - `uv run python -m unittest tests/test_auth.py`
- Run a single test case:
  - `uv run python -m unittest tests.test_auth.AuthTests.test_valid_token`

API test client (exercises the OpenAI-compatible endpoints; requires server running):
- `uv run python tests/test_app.py --base-url http://127.0.0.1:8001 --api-key local-dev --voice af_bella`

## Runtime Configuration
- Env vars (defaults in `src/mlx_openai_tts/config.py`):
  - `HOST`, `PORT` for uvicorn.
  - `API_KEY` to enable auth (Bearer token).
  - `TTS_MODELS_JSON` path (defaults to `models.json`).
  - `TTS_MAX_CHARS`, `TTS_WARMUP_TEXT`, `TTS_MLX_STRICT`.
- Audio formats `mp3`, `opus`, `aac`, `flac` require `ffmpeg` via Homebrew.

## Code Style and Conventions
General:
- Python only. 4-space indent. Keep lines <= 100 chars.
- Use `from __future__ import annotations` at the top of Python files.
- Use double quotes for strings (Ruff configured).
- Prefer explicit typing; keep type hints consistent with Pyright (basic mode).

Imports:
- Group imports in this order: standard library, third-party, local package.
- Use absolute imports from `mlx_openai_tts` (avoid relative imports outside package).
- Favor `collections.abc` for typing (`Mapping`, `Iterable`, etc.).

Naming:
- `snake_case` for functions/variables.
- `PascalCase` for classes, including Pydantic models.
- `UPPER_SNAKE_CASE` for constants.

Formatting:
- Use Ruff formatter (`ruff format .`).
- Ruff lint rules enabled: `E`, `F`, `I`, `B`, `UP` with `E501` ignored.
- Keep docstrings minimal; add comments only when behavior is non-obvious.

Types and Pydantic:
- Use Pydantic `BaseModel` for request/response schemas in `schemas.py`.
- Use `Literal` for constrained string enums (e.g., response formats).
- Prefer `Field` for defaults and validation constraints.
- Use dataclasses for simple immutable runtime config objects when appropriate.

Error Handling Patterns:
- Internal validation failures raise `RuntimeError` with clear messages.
- API layer converts internal failures to `fastapi.HTTPException` with 400/401 codes.
- Chain exceptions with `raise ... from exc` to preserve context.
- Validate inputs early (e.g., model IDs, voice IDs, max input length).

Concurrency and Streaming:
- Inference is serialized with a lock (`infer_lock`) for predictable latency.
- Streaming uses `StreamingResponse` and helper functions in `audio.py`.
- Keep headers consistent (`X-Model`, `X-Voice`, `X-Sample-Rate`, `X-Latency-Ms`).

## Files and Responsibilities
- `server.py`: FastAPI app, endpoints, auth, model loading, and streaming paths.
- `auth.py`: Bearer token validation.
- `registry.py`: `models.json` schema parsing and validation.
- `config.py`: Env parsing and defaults.
- `schemas.py`: Pydantic request/response models.
- `audio.py`/`engine.py`: Audio formatting and MLX model management.

## Testing Notes
- Tests are written with `unittest` (no pytest).
- Keep each test focused on one behavior (auth, registry validation, API client).
- The API test client is integration-like and expects a running server.

## PR/Commit Hygiene
- Keep changes scoped and avoid unrelated refactors.
- Mention test commands and results in PRs or change notes.
- No enforced commit style; use concise imperative messages.

## Agent Checklist
- Read relevant files before editing.
- Update tests or docs if behavior changes.
- Run formatter/lint if you change Python files.
- Prefer minimal, local changes over broad rewrites.
