"""
Demo: token-calibrator in action.

Showcases prior estimates, learning from observations, and how the model
adapts its coefficients.

Run: python examples/demo.py
"""

from src.calibrator import TokenCalibrator, estimate_tokens_from_priors


def main():
    print("=== token-calibrator demo (Python) ===\n")

    # 1. Stateless prior estimate
    english = "Hello, world! This is a test of the token calibrator."
    chinese = "你好世界，这是一个测试。"
    mixed = "Hello 你好 123 🎉"

    print("── Priors (no learning yet) ──")
    print(f"English  : {len(english):>4} chars → ~{estimate_tokens_from_priors(english):>3} tokens")
    print(f"Chinese  : {len(chinese):>4} chars → ~{estimate_tokens_from_priors(chinese):>3} tokens")
    print(f"Mixed    : {len(mixed):>4} chars → ~{estimate_tokens_from_priors(mixed):>3} tokens")

    # 2. Create a calibrator and learn
    cal = TokenCalibrator()
    print("\n── Before observing ──")
    print(f"English estimate: {cal.estimate(english)} tokens")
    print(f"Coefficients    : {cal.coefficients()}")

    # 3. Feed two observations
    cal.observe("Hello world", 3)
    cal.observe("你好世界", 6)

    print("\n── After observing 2 rounds ──")
    print(f"English estimate: {cal.estimate(english)} tokens (was {estimate_tokens_from_priors(english)})")
    print(f"Coefficients    : {cal.coefficients()}")

    # 4. Feed more data
    more_data = [
        ("short", 2),
        ("a bit longer english text here", 8),
        ("more english words for the model to learn from", 12),
        ("中文中文中文中文", 12),
        ("1234567890", 4),
    ]
    for text, tokens in more_data:
        cal.observe(text, tokens)

    print(f"\n── After {len(more_data)} more observations ──")
    print(f"English estimate: {cal.estimate(english)} tokens")
    print(f"Chinese estimate: {cal.estimate(chinese)} tokens")
    print(f"Coefficients    : {cal.coefficients()}")

    # 5. Snapshot round-trip
    snap = cal.snapshot()
    restored = TokenCalibrator(snapshot=snap)
    print(f"\n── Restored from snapshot ──")
    print(f"English estimate: {restored.estimate(english)} (match = {restored.estimate(english) == cal.estimate(english)})")

    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
