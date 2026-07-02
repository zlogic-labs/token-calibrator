"""
token-calibrator demo

Demonstrates two usage modes:
  1. Calibrate mode  — feed sample observations and export the accumulator
  2. Estimate mode — load calibrated data (or use built-in models) and estimate

Usage:
    python examples/demo.py calibrate               # calibrate & export to calibrated-snapshot.json
    python examples/demo.py estimate                # estimate using built-in models
    python examples/demo.py estimate calibrated-snapshot.json  # use custom snapshot
"""

import json
import sys
import os

from token_calibrator import (
    TokenCalibrator,
    TokenEstimator,
    TOKEN_BUCKET_PRIORS,
    BUILTIN_TOKEN_RATES,
    classify_token_buckets,
    estimate_tokens,
)


# ────────────────────── Calibrate mode ──────────────────────

def cmd_calibrate() -> None:
    print("=== Calibrate mode ===\n")

    # Create a calibrator with a small prior strength so data dominates quickly.
    cal = TokenCalibrator({"priorStrength": 1_000})

    # Simulated real usage observations (prompt text, actual token count from API).
    observations: list[tuple[str, int]] = [
        ("Hello world", 3),
        ("The quick brown fox jumps over the lazy dog", 10),
        ("你好，世界", 6),
        ("안녕하세요", 8),
        ("Привет мир", 5),
        ("123 456 7890", 6),
        ("🚀 Token estimation is amazing! 🎉", 12),
        ("Mixed 你好 Hello 123 😊", 9),
    ]

    for text, tokens in observations:
        counts = classify_token_buckets(text)
        print(f"  observe: {counts}  →  {tokens} tokens  ({json.dumps(text)})")
        cal.observe(text, tokens)

    # Show learned rates.
    rates = cal.rates()
    print("\nLearned per-bucket rates:")
    for bucket, rate in sorted(rates.items()):
        print(f"  {bucket:<10} {rate:.4f}")

    # Estimate a few samples.
    print("\nEstimates after calibration:")
    for text in ["Hello world", "你好", "Mixed 你好 123 😊"]:
        print(f"  {json.dumps(text):<30} → {cal.estimate(text)} tokens")

    # Export accumulator to file.
    matrix = cal.to_matrix()
    snapshot = {"models": {"demo-model": matrix}}
    with open("calibrated-snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)
    print("\nExported accumulator to calibrated-snapshot.json\n")


# ───────────────────── Estimate mode ─────────────────────

def cmd_estimate(snapshot_path: str | None = None) -> None:
    print("=== Estimate mode ===\n")

    matrices: dict[str, dict] = {}

    if snapshot_path and os.path.exists(snapshot_path):
        with open(snapshot_path) as f:
            raw = json.load(f)
        matrices = raw.get("models", raw)
        print(f"Loaded {len(matrices)} model(s) from {snapshot_path}\n")
    else:
        print("Using built-in default models (no snapshot file provided)\n")

    est = TokenEstimator(matrices)

    # Test texts covering different scripts.
    test_texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "你好，世界",
        "안녕하세요",
        "Привет мир",
        "123 456 7890",
        "🚀 Token estimation is amazing! 🎉",
    ]

    # A few model names to test.
    model_names = list(BUILTIN_TOKEN_RATES.keys())[:5]
    model_names.append("demo-model")  # may or may not be present

    for model in model_names:
        print(f"── {model} ──")
        if est.has(model):
            print("  (user-calibrated data loaded)")
        for text in test_texts:
            t = est.estimate(model, text)
            print(f"  {json.dumps(text):<50} → {t} tokens")
        print()

    # Also show what falls back to bare priors.
    print("── unknown-model (falls back to prior) ──")
    for text in ["Hello", "你好"]:
        t = est.estimate("unknown-model", text)
        print(f"  {json.dumps(text):<50} → {t} tokens")


# ─────────────────────── Main ───────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "estimate"
    arg = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"token-calibrator demo (mode: {mode})\n")

    if mode == "calibrate":
        cmd_calibrate()
    else:
        cmd_estimate(arg)
