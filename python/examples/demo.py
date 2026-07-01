"""
Demo: token-calibrator in action — loading from models.json.

Shows how to initialise from a model snapshot file, then learn and
export your own snapshot.

Run: python examples/demo.py (from the python/ directory)
"""

import json
import os
from src.calibrator import TokenCalibrator, estimate_tokens_from_priors


def main():
    print("=== token-calibrator demo (Python) ===\n")

    # Resolve path to models/models.json relative to this file
    demo_dir = os.path.dirname(__file__)
    models_path = os.path.join(demo_dir, "..", "..", "models", "models.json")
    with open(models_path) as f:
        models_json = f.read()

    english = "Hello, world! This is a test of the token calibrator."
    chinese = "你好世界，这是一个测试。"
    mixed = "Hello 你好 123 🎉"

    # 1. Load from models.json
    print("── Load from models.json ──")
    cal = TokenCalibrator.from_model("gpt-4o", models_json)
    # from_model returns a fresh prior-seeded calibrator if model not found
    print("Loaded model: gpt-4o")
    print(f"English estimate: {cal.estimate(english)} tokens")
    print(f"Coefficients    : {cal.coefficients()}")

    # 2. Compare with stateless priors
    print("\n── Stateless priors (no calibration) ──")
    print(f"English  : {len(english):>4} chars → ~{estimate_tokens_from_priors(english):>3} tokens")
    print(f"Chinese  : {len(chinese):>4} chars → ~{estimate_tokens_from_priors(chinese):>3} tokens")
    print(f"Mixed    : {len(mixed):>4} chars → ~{estimate_tokens_from_priors(mixed):>3} tokens")

    # 3. Feed observations
    print("\n── Training with real observations ──")
    cal.observe("Hello world", 3)
    cal.observe("你好世界", 6)

    more_data = [
        ("short", 2),
        ("a bit longer english text here", 8),
        ("more english words for the model to learn from", 12),
        ("中文中文中文中文", 12),
        ("1234567890", 4),
    ]
    for text, tokens in more_data:
        cal.observe(text, tokens)

    print(f"After {2 + len(more_data)} observations:")
    print(f"English estimate: {cal.estimate(english)} tokens")
    print(f"Chinese estimate: {cal.estimate(chinese)} tokens")
    print(f"Coefficients    : {cal.coefficients()}")

    # 4. Snapshot round-trip — load fresh from model and replay training
    print("\n── Restored from fresh model + same training ──")
    restored = TokenCalibrator.from_model("gpt-4o", models_json)
    restored.observe("Hello world", 3)
    restored.observe("你好世界", 6)
    for text, tokens in more_data:
        restored.observe(text, tokens)

    print(f"English estimate: {restored.estimate(english)} (match = {restored.estimate(english) == cal.estimate(english)})")

    # 5. Export your trained snapshot (to contribute back!)
    print("\n── Your trained snapshot (ready to contribute!) ──")
    trained = cal.snapshot()
    print(f"a: {trained['a']}")
    print(f"g: {trained['g']}")
    print(f"strength: {trained['strength']}")

    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
