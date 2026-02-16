#!/usr/bin/env python3
"""
Stress test script for WTF transcription servers.

Compares processing times between local Whisper and MLX Whisper backends.
Targets the OpenAI-compatible endpoint: POST /v1/audio/transcriptions

Usage:
  # Compare local whisper (port 9001) vs MLX (port 8000)
  uv run python scripts/stress_test.py \
    --local-whisper http://localhost:9001 \
    --mlx http://localhost:8000 \
    --iterations 10

  # Test MLX only with different models
  uv run python scripts/stress_test.py \
    --mlx http://localhost:8000 \
    --models mlx-community/whisper-tiny mlx-community/whisper-turbo \
    --iterations 5

  # Use a real audio file
  uv run python scripts/stress_test.py \
    --local-whisper http://localhost:9001 \
    --mlx http://localhost:8000 \
    --audio recording.wav
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import time
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    print("Install httpx: uv add httpx (or use dev deps: uv sync --all-extras)", file=sys.stderr)
    sys.exit(1)


def make_sample_wav(duration_sec: float = 5.0) -> bytes:
    """Generate a minimal valid WAV file (silence at 16kHz, mono, 16-bit)."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    wav_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return wav_header + b"\x00" * data_size


def get_content_type(filename: str) -> str:
    """Map filename extension to audio content type."""
    ext = Path(filename).suffix.lower()
    mime = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/m4a",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".webm": "audio/webm",
    }
    return mime.get(ext, "audio/wav")


def transcribe(
    base_url: str,
    audio_bytes: bytes,
    filename: str = "audio.wav",
    model: str | None = None,
    timeout: int = 600,
) -> tuple[float, dict[str, Any] | None, str | None]:
    """
    POST audio to /v1/audio/transcriptions. Returns (elapsed_sec, response_json, error).
    """
    url = base_url.rstrip("/") + "/v1/audio/transcriptions"
    content_type = get_content_type(filename)
    files = {"file": (filename, io.BytesIO(audio_bytes), content_type)}
    data: dict[str, Any] = {"response_format": "verbose_json"}
    if model:
        data["model"] = model

    start = time.monotonic()
    try:
        resp = httpx.post(url, files=files, data=data, timeout=timeout)
        elapsed = time.monotonic() - start
        if resp.status_code != 200:
            return elapsed, None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        return elapsed, resp.json(), None
    except httpx.HTTPError as e:
        elapsed = time.monotonic() - start
        return elapsed, None, str(e)


def percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile (0-100) of sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stress test WTF transcription servers (local Whisper vs MLX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--local-whisper",
        metavar="URL",
        help="Local Whisper server URL (e.g. http://localhost:9001)",
    )
    parser.add_argument(
        "--mlx",
        metavar="URL",
        help="MLX Whisper server URL (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="MLX models to test (e.g. mlx-community/whisper-tiny mlx-community/whisper-turbo). "
        "Used with --mlx. Each model becomes a separate target.",
    )
    parser.add_argument(
        "--audio",
        metavar="FILE",
        type=Path,
        help="Path to audio file (WAV, MP3, etc). If omitted, generates 5s silence.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration in seconds for generated sample audio (default: 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per target (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup requests before timing (default: 1)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent requests (default: 1). Use 1 for stable per-request timing.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    if not args.local_whisper and not args.mlx:
        parser.error("Specify at least one of --local-whisper or --mlx")

    targets: list[tuple[str, str, str | None]] = []
    if args.local_whisper:
        targets.append(("Local Whisper", args.local_whisper, None))
    if args.mlx:
        if args.models:
            for m in args.models:
                short = m.split("/")[-1] if "/" in m else m
                targets.append((f"MLX ({short})", args.mlx, m))
        else:
            targets.append(("MLX", args.mlx, "mlx-community/whisper-turbo"))

    # Load or generate audio
    if args.audio:
        if not args.audio.exists():
            print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
            return 1
        audio_bytes = args.audio.read_bytes()
        filename = args.audio.name
    else:
        audio_bytes = make_sample_wav(args.duration)
        filename = "audio.wav"

    run_stress_test_v2(targets, audio_bytes, filename, args.iterations, args.warmup, args.timeout)
    return 0


def run_stress_test_v2(
    targets: list[tuple[str, str, str | None]],
    audio_bytes: bytes,
    filename: str,
    iterations: int,
    warmup: int,
    timeout: int,
) -> None:
    """Run stress test with (label, base_url, model) per target."""
    results: dict[str, list[float]] = {label: [] for label, _, _ in targets}

    for label, base_url, model in targets:
        for i in range(warmup + iterations):
            elapsed, data, err = transcribe(base_url, audio_bytes, filename, model, timeout)
            if err:
                print(f"  [{label}] iter {i}: {err}", file=sys.stderr)
                continue
            if i >= warmup:
                results[label].append(elapsed)

    # Report
    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)
    print(f"Audio size: {len(audio_bytes):,} bytes")
    print(f"Iterations: {iterations} (warmup: {warmup})")
    print()

    for label, base_url, model in targets:
        times = results[label]
        if not times:
            print(f"{label}: NO SUCCESSFUL REQUESTS")
            continue

        times_sorted = sorted(times)
        n = len(times_sorted)
        avg = sum(times_sorted) / n
        p50 = percentile(times_sorted, 50)
        p95 = percentile(times_sorted, 95)
        p99 = percentile(times_sorted, 99)

        print(f"{label}")
        print(f"  Requests: {n} successful")
        print(f"  Min:      {min(times_sorted):.2f}s")
        print(f"  Max:      {max(times_sorted):.2f}s")
        print(f"  Avg:      {avg:.2f}s")
        print(f"  P50:      {p50:.2f}s")
        print(f"  P95:      {p95:.2f}s")
        print(f"  P99:      {p99:.2f}s")
        print()


if __name__ == "__main__":
    sys.exit(main())
