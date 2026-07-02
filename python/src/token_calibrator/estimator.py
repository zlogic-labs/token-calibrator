"""
TokenEstimator (reader) — the stateful consumer side (renderer / CLI display).
Built from a `{model: matrix}` map (as produced by calibrators and merged by
object spread); derives + caches each model's rates up front. Pure math is in
`calibration.py`; this is per-model lookup plus the shipped-baseline fallback.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .buckets import estimate_tokens, TOKEN_BUCKET_PRIORS, TokenRates
from .calibration import is_valid_accumulator, rates_from_accumulator, TokenAccumulator
from .builtin_rates import BUILTIN_TOKEN_RATES

def _get_opt(opts: Any, key: str, default: Any = None) -> Any:
    """Extract an option from either a dict or an object."""
    if opts is None:
        return default
    if isinstance(opts, dict):
        return opts.get(key, default)
    return getattr(opts, key, default)


class TokenEstimatorOptions:
    priorStrength: Optional[float] = None
    baseline: Optional[Dict[str, TokenRates]] = None
    prior: Optional[TokenRates] = None


class TokenEstimator:
    """
    Read-only, per-model estimator. Three-tier fallback per model:

    user-calibrated rates → shipped baseline (BUILTIN_TOKEN_RATES) → prior

    The baseline also acts as the ridge prior when a model DOES have user
    data, so a lightly-calibrated model stays anchored near its shipped rates.

    Usage:
        est = TokenEstimator(loaded_matrices)
        tokens = est.estimate("gpt-5.5", request_body)
    """

    def __init__(
        self,
        matrices: Optional[Dict[str, TokenAccumulator]] = None,
        opts: Any = None,
    ) -> None:
        self._rates_by_model: Dict[str, TokenRates] = {}

        bl = _get_opt(opts, "baseline")
        self._baseline: Dict[str, TokenRates] = bl if bl is not None else BUILTIN_TOKEN_RATES

        pr = _get_opt(opts, "prior")
        self._prior: TokenRates = pr if pr is not None else TOKEN_BUCKET_PRIORS

        self._prior_strength: Optional[float] = _get_opt(opts, "priorStrength")

        if matrices is not None:
            for model, acc in matrices.items():
                if not is_valid_accumulator(acc):
                    continue
                self._rates_by_model[model] = rates_from_accumulator(acc, {
                    "priorStrength": self._prior_strength,
                    "prior": self._baseline.get(model, self._prior),
                })

    def estimate(self, model: str, input_str: str) -> int:
        """Estimated tokens for `input` under `model`'s effective rates."""
        return estimate_tokens(input_str, self.rates(model))

    def rates(self, model: str) -> TokenRates:
        """Effective rates for `model`: user-calibrated → baseline → prior."""
        if model in self._rates_by_model:
            return self._rates_by_model[model]
        if model in self._baseline:
            return self._baseline[model]
        return self._prior

    def has(self, model: str) -> bool:
        """Whether this model has its OWN calibrated (user-data) rates."""
        return model in self._rates_by_model
