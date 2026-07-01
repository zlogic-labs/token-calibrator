"""Tests for token-calibrator."""

import math
import pytest
from src.calibrator import (
    TokenCalibrator,
    TokenCalibratorOptions,
    classify_token_buckets,
    estimate_tokens_from_priors,
    solve_linear_system,
    is_valid_snapshot,
)


def test_classify_latin():
    c = classify_token_buckets("Hello")
    assert c == {"han": 0, "latin": 5, "digit": 0, "other": 0}


def test_classify_chinese():
    c = classify_token_buckets("你好")
    assert c == {"han": 2, "latin": 0, "digit": 0, "other": 0}


def test_classify_mixed():
    c = classify_token_buckets("Hello 世界 123 😊")
    # H e l l o + 3 spaces = 8 Latin; 世 界 = 2 Han; 1 2 3 = 3 Digit; 😊 = 4 bytes Other
    assert c["han"] == 2
    assert c["latin"] == 8
    assert c["digit"] == 3
    assert c["other"] == 4


def test_classify_empty():
    c = classify_token_buckets("")
    assert c == {"han": 0, "latin": 0, "digit": 0, "other": 0}


def test_estimate_from_priors():
    assert estimate_tokens_from_priors("Hello world") == 3


def test_estimate_from_priors_empty():
    assert estimate_tokens_from_priors("") == 0


def test_calibrator_prior_estimate():
    cal = TokenCalibrator()
    assert cal.estimate("Hello world") == 3


def test_calibrator_adapts():
    cal = TokenCalibrator(TokenCalibratorOptions(prior_strength=1.0))
    for _ in range(20):
        cal.observe("Hello world", 4)
    assert cal.estimate("Hello world") == 4


def test_calibrator_snapshot_roundtrip():
    cal = TokenCalibrator()
    cal.observe("test", 2)
    snap = cal.snapshot()
    restored = TokenCalibrator(snapshot=snap)
    assert restored.estimate("test") == cal.estimate("test")


def test_coefficients():
    cal = TokenCalibrator()
    coef = cal.coefficients()
    for bucket in ("han", "latin", "digit", "other"):
        assert bucket in coef
        assert math.isfinite(coef[bucket])


def test_solve_linear_system():
    # Identity matrix: A·x = b => x = b
    a = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    b = [2.0, 3.0, 4.0, 5.0]
    x = solve_linear_system(a, b)
    assert x == pytest.approx([2.0, 3.0, 4.0, 5.0])


def test_is_valid_snapshot():
    assert not is_valid_snapshot(None)
    assert not is_valid_snapshot({})
    snap = {
        "a": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "g": [1, 2, 3, 4],
        "strength": 1000.0,
    }
    assert is_valid_snapshot(snap)


def test_from_model():
    json_data = """
    {
        "models": {
            "test-model": {
                "a": [[1000,0,0,0],[0,1000,0,0],[0,0,1000,0],[0,0,0,1000]],
                "g": [1000,250,400,600],
                "strength": 1000
            }
        }
    }
    """
    cal = TokenCalibrator.from_model("test-model", json_data)
    assert cal.estimate("Hello world") == 3


def test_from_model_missing():
    cal = TokenCalibrator.from_model("nonexistent", '{"models": {}}')
    assert cal.estimate("Hello world") == 3  # falls back to fresh calibrator
