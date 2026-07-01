"""Token calibrator — self-learning token estimator for LLM context-size display.

Uses online ridge regression to learn per-bucket token rates (Han, Latin, Digit,
Other) from real usage observations. No bundled tokenizer required.
"""

from .calibrator import (
    TokenCalibrator,
    TokenCalibratorOptions,
    TokenCalibratorSnapshot,
    classify_token_buckets,
    estimate_tokens_from_priors,
    is_valid_snapshot,
    solve_linear_system,
)

__all__ = [
    "TokenCalibrator",
    "TokenCalibratorOptions",
    "TokenCalibratorSnapshot",
    "classify_token_buckets",
    "estimate_tokens_from_priors",
    "is_valid_snapshot",
    "solve_linear_system",
]

__version__ = "1.0.0"
