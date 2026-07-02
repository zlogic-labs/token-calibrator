"""Token calibrator — self-learning token estimator for LLM context-size display.

Uses online ridge regression to learn per-bucket token rates from real usage
observations. No bundled tokenizer required.
"""

from .buckets import (
    TOKEN_BUCKETS,
    N_BUCKETS,
    TokenBucket,
    TokenRates,
    TOKEN_BUCKET_PRIORS,
    classify_token_buckets,
    feature_vector,
    estimate_tokens,
)
from .calibration import (
    TokenAccumulator,
    AccumulateOptions,
    RatesOptions,
    DEFAULT_PRIOR_STRENGTH,
    empty_accumulator,
    accumulate,
    rates_from_accumulator,
    derive_rates,
    is_valid_accumulator,
    solve_linear_system,
)
from .calibrator import TokenCalibrator, TokenCalibratorOptions
from .estimator import TokenEstimator, TokenEstimatorOptions
from .builtin_rates import BUILTIN_TOKEN_RATES

__all__ = [
    # Buckets
    "TOKEN_BUCKETS",
    "N_BUCKETS",
    "TokenBucket",
    "TokenRates",
    "TOKEN_BUCKET_PRIORS",
    "classify_token_buckets",
    "feature_vector",
    "estimate_tokens",
    # Calibration
    "TokenAccumulator",
    "AccumulateOptions",
    "RatesOptions",
    "DEFAULT_PRIOR_STRENGTH",
    "empty_accumulator",
    "accumulate",
    "rates_from_accumulator",
    "derive_rates",
    "is_valid_accumulator",
    "solve_linear_system",
    # Calibrator
    "TokenCalibrator",
    "TokenCalibratorOptions",
    # Estimator
    "TokenEstimator",
    "TokenEstimatorOptions",
    # Builtin
    "BUILTIN_TOKEN_RATES",
]

__version__ = "1.0.0"
