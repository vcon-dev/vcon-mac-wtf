# vcon-mac-wtf

MLX Whisper transcription server for Apple Silicon with WTF/vCon support.

Runs OpenAI's Whisper speech-to-text model locally on Apple Silicon (M1/M2/M3/M4) using the [MLX](https://github.com/ml-explore/mlx) framework for GPU-accelerated inference. Outputs transcriptions in [World Transcription Format (WTF)](https://datatracker.ietf.org/doc/draft-howe-vcon-wtf-extension/) and can enrich [vCon](https://datatracker.ietf.org/doc/draft-ietf-vcon-vcon-core/) documents with transcription analysis.

## Features

- GPU-accelerated Whisper inference on Apple Silicon via MLX
- OpenAI-compatible API (`POST /v1/audio/transcriptions`) — drop-in replacement
- vCon-native API (`POST /transcribe`) — accepts a vCon, returns enriched vCon with WTF transcription
- Word-level timestamps
- Multiple model sizes (tiny through large-v3)
- Integrates with [wtf-server](https://github.com/vcon-dev/wtf-server) as the `mlx-whisper` provider

## Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- ffmpeg: `brew install ffmpeg`

## Quickstart

```bash
# Install (choose one)
pip install vcon-mac-wtf          # PyPI
brew tap thomashowe/vcon && brew install vcon-mac-wtf  # Homebrew (after adding tap)

# Or from source
uv sync --all-extras

# Start the server (downloads model on first run)
vcon-mac-wtf
# or: make run
# or: uv run uvicorn vcon_mac_wtf.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```

### Transcribe Audio (OpenAI-compatible)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=mlx-community/whisper-turbo" \
  -F "response_format=verbose_json"
```

Response formats: `json`, `text`, `verbose_json`, `wtf`

### Transcribe vCon

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d @my_vcon.json
```

Returns the vCon with WTF transcription analysis appended.

### List Models

```bash
curl http://localhost:8000/v1/models
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `info` | Logging level |
| `MLX_MODEL` | `mlx-community/whisper-turbo` | Default Whisper model |
| `PRELOAD_MODEL` | `true` | Load model at startup |
| `MAX_AUDIO_SIZE_MB` | `100` | Max upload size |
| `HF_TOKEN` | - | HuggingFace token for faster model downloads (optional) |

Copy `.env.example` to `.env` to customize. Add `HF_TOKEN=hf_xxx` to `.env` before first run to speed up model downloads and avoid rate limits.

## Available Models

| Short Name | Model ID | Parameters |
|---|---|---|
| `tiny` | `mlx-community/whisper-tiny` | 39M |
| `base` | `mlx-community/whisper-base` | 74M |
| `small` | `mlx-community/whisper-small` | 244M |
| `medium` | `mlx-community/whisper-medium` | 769M |
| `large-v3` | `mlx-community/whisper-large-v3` | 1.55B |
| `turbo` | `mlx-community/whisper-turbo` | 809M |

## Integration with wtf-server

This server works as a provider for the existing TypeScript wtf-server. Set in the wtf-server `.env`:

```
ASR_PROVIDER=mlx-whisper
MLX_WHISPER_URL=http://localhost:8000
MLX_WHISPER_MODEL=mlx-community/whisper-turbo
```

## Installation

| Method | Command |
|--------|---------|
| PyPI | `pip install vcon-mac-wtf` |
| Homebrew | `brew tap thomashowe/vcon && brew install vcon-mac-wtf` |
| Source | `uv sync --all-extras` |

## Publishing

**PyPI:** `python -m build && twine upload dist/*`

**Homebrew:** Formula is in `homebrew/`. Add to a tap repo (e.g. `homebrew-vcon`) at `Formula/vcon-mac-wtf.rb`. See `homebrew/README.md`.

## Development

```bash
make install    # Install all deps including dev
make dev        # Run with auto-reload
make test       # Run unit tests
make test-all   # Run all tests including integration
make lint       # Lint with ruff
make format     # Format with ruff
```

## Testing

```bash
# Unit tests (no MLX required, uses mocks)
make test

# Integration tests (requires Apple Silicon + MLX)
make test-all

# Coverage
make test-cov
```
