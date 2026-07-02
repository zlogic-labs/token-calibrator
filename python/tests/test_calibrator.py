"""Tests for token-calibrator.

These tests mirror the same logic across TypeScript, Python, Rust, and Go.
"""

import math
import pytest
from token_calibrator import (
    TOKEN_BUCKETS,
    N_BUCKETS,
    TOKEN_BUCKET_PRIORS,
    TokenCalibrator,
    TokenEstimator,
    TokenAccumulator,
    AccumulateOptions,
    RatesOptions,
    DEFAULT_PRIOR_STRENGTH,
    classify_token_buckets,
    feature_vector,
    estimate_tokens,
    empty_accumulator,
    accumulate,
    rates_from_accumulator,
    derive_rates,
    is_valid_accumulator,
    solve_linear_system,
    BUILTIN_TOKEN_RATES,
)


# ---------------------------------------------------------------------------
# classify_token_buckets
# ---------------------------------------------------------------------------

class TestClassifyTokenBuckets:
    def test_latin_text(self):
        c = classify_token_buckets("Hello")
        assert c["han"] == 0
        assert c["latin"] == 5
        assert c["digit"] == 0
        assert c["hangul"] == 0
        assert c["cyrillic"] == 0
        assert c["emoji"] == 0
        assert c["other"] == 0

    def test_chinese_text(self):
        c = classify_token_buckets("你好")
        assert c["han"] == 2
        assert c["latin"] == 0
        assert c["other"] == 0

    def test_mixed_text(self):
        # "Hello 世界 123 😊"
        s = "Hello 世界 123 \U0001f60a"
        c = classify_token_buckets(s)
        assert c["han"] == 2   # 世界
        assert c["latin"] == 8  # Hello + space + space + space
        assert c["digit"] == 3
        assert c["emoji"] == 1  # 😊
        # The characters not in han/latin/digit/emoji: none in this string
        # Spaces are Latin (0x20), so other = 0

    def test_empty_string(self):
        c = classify_token_buckets("")
        for b in TOKEN_BUCKETS:
            assert c[b] == 0

    def test_korean_text(self):
        c = classify_token_buckets("안녕하세요")
        assert c["hangul"] == 5
        assert c["other"] == 0

    def test_cyrillic_text(self):
        c = classify_token_buckets("Привет")
        assert c["cyrillic"] == 6
        assert c["other"] == 0

    def test_emoji_text(self):
        c = classify_token_buckets("🚀🎉")
        assert c["emoji"] == 2
        assert c["other"] == 0


# ---------------------------------------------------------------------------
# feature_vector
# ---------------------------------------------------------------------------

class TestFeatureVector:
    def test_order_matches_buckets(self):
        counts = {"han": 2, "latin": 5, "digit": 3, "hangul": 0, "cyrillic": 0, "emoji": 1, "other": 4}
        fv = feature_vector(counts)
        assert fv == [2.0, 5.0, 3.0, 0.0, 0.0, 1.0, 4.0]


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_prior_estimate_hello_world(self):
        # "Hello world" = 11 chars → 11 * 0.25 latin = 2.75 → round 3
        assert estimate_tokens("Hello world", TOKEN_BUCKET_PRIORS) == 3

    def test_empty(self):
        assert estimate_tokens("", TOKEN_BUCKET_PRIORS) == 0

    def test_floor_at_1(self):
        # Single char with low rate: 1 * 0.25 = 0.25 → round 0 → max(1)=1
        assert estimate_tokens("a", TOKEN_BUCKET_PRIORS) == 1


# ---------------------------------------------------------------------------
# empty_accumulator / is_valid_accumulator
# ---------------------------------------------------------------------------

class TestAccumulatorValidation:
    def test_empty_is_valid(self):
        acc = empty_accumulator()
        assert is_valid_accumulator(acc)

    def test_none_is_invalid(self):
        assert not is_valid_accumulator(None)

    def test_wrong_type_is_invalid(self):
        assert not is_valid_accumulator("hello")
        assert not is_valid_accumulator(42)

    def test_nan_is_invalid(self):
        acc = empty_accumulator()
        acc["a"][0][0] = float("nan")
        assert not is_valid_accumulator(acc)

    def test_inf_is_invalid(self):
        acc = empty_accumulator()
        acc["g"][0] = float("inf")
        assert not is_valid_accumulator(acc)


# ---------------------------------------------------------------------------
# accumulate / rates_from_accumulator
# ---------------------------------------------------------------------------

class TestAccumulate:
    def test_non_positive_tokens_ignored(self):
        acc = empty_accumulator()
        acc2 = accumulate(acc, "Hello", 0)
        assert acc2 is acc  # returned unchanged

    def test_zero_features_ignored(self):
        acc = empty_accumulator()
        acc2 = accumulate(acc, "", 10)
        assert acc2 is acc  # returned unchanged

    def test_accumulate_and_solve_identity(self):
        # After observing "Hello world" with 4 real tokens * 20 times
        # with strength=1, the latin rate should be close to 4/11
        acc = empty_accumulator()
        for _ in range(20):
            acc = accumulate(acc, "Hello world", 4)
        rates = rates_from_accumulator(acc, {"priorStrength": 1.0})
        # "Hello world" has 11 latin chars
        # Prior latin rate is 0.25, strength=1
        # After 20 obs of 4 tokens, dominated by data: ~4/11
        assert abs(rates["latin"] - 4.0 / 11.0) < 0.1

    def test_forgetting(self):
        acc = empty_accumulator()
        acc = accumulate(acc, "a", 10, {"forgetting": 0.5})
        # With forgetting, older data decays
        rates_before = rates_from_accumulator(acc, {"priorStrength": 1.0})
        assert rates_before["latin"] > 0


# ---------------------------------------------------------------------------
# solve_linear_system
# ---------------------------------------------------------------------------

class TestSolveLinearSystem:
    def test_identity(self):
        a = [[1.0, 0, 0, 0, 0, 0, 0],
             [0, 1.0, 0, 0, 0, 0, 0],
             [0, 0, 1.0, 0, 0, 0, 0],
             [0, 0, 0, 1.0, 0, 0, 0],
             [0, 0, 0, 0, 1.0, 0, 0],
             [0, 0, 0, 0, 0, 1.0, 0],
             [0, 0, 0, 0, 0, 0, 1.0]]
        b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        x = solve_linear_system(a, b)
        assert x == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


# ---------------------------------------------------------------------------
# TokenCalibrator
# ---------------------------------------------------------------------------

class TestTokenCalibrator:
    def test_starts_with_prior_rates(self):
        cal = TokenCalibrator({"priorStrength": 1.0})
        r = cal.rates()
        assert r["han"] == pytest.approx(1.0, abs=1e-9)
        assert r["latin"] == pytest.approx(0.25, abs=1e-9)
        assert r["digit"] == pytest.approx(0.4, abs=1e-9)

    def test_adapts_after_observations(self):
        cal = TokenCalibrator({"priorStrength": 1.0})
        for _ in range(20):
            cal.observe("Hello world", 4)
        r = cal.rates()
        assert r["latin"] == pytest.approx(4.0 / 11.0, abs=0.1)

    def test_estimate_from_prior(self):
        cal = TokenCalibrator({"priorStrength": 1.0})
        assert cal.estimate("Hello world") == 3

    def test_estimate_after_training(self):
        cal = TokenCalibrator({"priorStrength": 1.0})
        for _ in range(20):
            cal.observe("Hello world", 4)
        assert cal.estimate("Hello world") == 4

    def test_round_trip_matrix(self):
        cal = TokenCalibrator()
        cal.observe("test", 2)
        matrix = cal.to_matrix()
        restored = TokenCalibrator({"accumulator": matrix})
        assert restored.estimate("test") == cal.estimate("test")
        assert restored.rates() == cal.rates()

    def test_corrupt_matrix_ignored(self):
        bad = {"a": [[1.0]], "g": [1.0]}  # wrong size
        cal = TokenCalibrator({"accumulator": bad})  # type: ignore[arg-type]
        # Falls back to empty
        assert is_valid_accumulator(cal.to_matrix())


# ---------------------------------------------------------------------------
# TokenEstimator
# ---------------------------------------------------------------------------

class TestTokenEstimator:
    def test_unknown_model_falls_back_to_baseline(self):
        est = TokenEstimator()
        # If baseline has the model, use it
        if "gpt-4o" in BUILTIN_TOKEN_RATES:
            t = est.estimate("gpt-4o", "Hello world")
            assert t > 0  # should work

    def test_estimate_from_registered_model(self):
        acc = empty_accumulator()
        for _ in range(20):
            acc = accumulate(acc, "Hello world", 4)
        est = TokenEstimator({"test-model": acc}, {"priorStrength": 1.0})
        assert est.estimate("test-model", "Hello world") == 4

    def test_empty_input(self):
        acc = empty_accumulator()
        est = TokenEstimator({"m": acc})
        assert est.estimate("m", "") == 0

    def test_has_model(self):
        acc = empty_accumulator()
        est = TokenEstimator({"m": acc})
        assert est.has("m")
        assert not est.has("nonexistent")

    def test_rates_fallback_tier(self):
        # No user data, no baseline → prior
        est = TokenEstimator()
        r = est.rates("nonexistent-model-xyz")
        assert r == TOKEN_BUCKET_PRIORS
