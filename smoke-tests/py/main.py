"""
Smoke test: token-calibrator (PyPI package)
Run: pip install token-calibrator && python main.py
"""
from token_calibrator import (
    TokenCalibrator, TokenEstimator,
    classify_token_buckets, estimate_tokens,
    TOKEN_BUCKET_PRIORS, BUILTIN_TOKEN_RATES,
)

# 1. Basic classify
c = classify_token_buckets("Hello 世界 123")
print(f"[PY] classifyTokenBuckets: {c}")

# 2. Prior estimate
t1 = estimate_tokens("Hello world", TOKEN_BUCKET_PRIORS)
print(f"[PY] estimate (prior): {t1} (expected 3)")

# 3. Calibrate + estimate
cal = TokenCalibrator({"priorStrength": 1_000})
observations = [
    ("Hello world", 3),
    ("你好世界", 6),
    ("12345", 5),
]
for text, tokens in observations:
    cal.observe(text, tokens)
t2 = cal.estimate("Hello world")
print(f"[PY] calibrate + estimate: {t2} (expected ~3)")

# 4. Estimator with built-in models
est = TokenEstimator()
model_names = list(BUILTIN_TOKEN_RATES.keys())[:3]
for model in model_names:
    t = est.estimate(model, "Hello world")
    print(f"[PY] {model}: {t} tokens")

# 5. Estimator with custom matrix
matrix = cal.to_matrix()
est2 = TokenEstimator({"my-model": matrix})
print(f"[PY] custom model estimate: {est2.estimate('my-model', 'Hello world')}")

print("[PY] All smoke tests passed!")
