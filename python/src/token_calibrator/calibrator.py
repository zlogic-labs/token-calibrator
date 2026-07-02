"""
TokenCalibrator (writer) — the stateful producer side. Wraps ONE model's
data-only accumulator: feed it real usage via `observe`, read back solved
`rates()` or the raw `to_matrix()` for persistence. All the math is the pure
`calibration.py` functions; this is just per-instance state + convenience.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .buckets import estimate_tokens, TokenRates
from .calibration import (
    accumulate,
    empty_accumulator,
    is_valid_accumulator,
    rates_from_accumulator,
    TokenAccumulator,
    AccumulateOptions,
    RatesOptions,
)


class TokenCalibratorOptions(RatesOptions, AccumulateOptions):
    accumulator: Optional[TokenAccumulator] = None


class TokenCalibrator:
    """
    Stateful writer around ONE model's accumulator.

    Usage:
        cal = TokenCalibrator()
        cal.observe(request_text, real_input_tokens)
        rates = cal.rates()  # for estimation
        matrix = cal.to_matrix()  # for persistence
    """

    def __init__(self, opts: Optional[TokenCalibratorOptions] = None) -> None:
        if opts is None:
            opts = {}
        self._forgetting: Optional[float] = opts.get("forgetting")
        self._prior_strength: Optional[float] = opts.get("priorStrength")
        self._prior: Optional[TokenRates] = opts.get("prior")

        acc = opts.get("accumulator")
        if acc is not None and is_valid_accumulator(acc):
            self._acc: TokenAccumulator = {
                "a": [row[:] for row in acc["a"]],
                "g": list(acc["g"]),
            }
        else:
            self._acc = empty_accumulator()

    def observe(self, input_str: str, real_tokens: float) -> None:
        """Fold in one observed round."""
        opts_accum: AccumulateOptions = {}
        if self._forgetting is not None:
            opts_accum["forgetting"] = self._forgetting
        self._acc = accumulate(self._acc, input_str, real_tokens, opts_accum)

    def rates(self) -> TokenRates:
        """Current solved per-bucket rates."""
        opts_rates: RatesOptions = {}
        if self._prior_strength is not None:
            opts_rates["priorStrength"] = self._prior_strength
        if self._prior is not None:
            opts_rates["prior"] = self._prior
        return rates_from_accumulator(self._acc, opts_rates)

    def estimate(self, input_str: str) -> int:
        """Convenience: estimate directly from this calibrator's current rates."""
        return estimate_tokens(input_str, self.rates())

    def to_matrix(self) -> TokenAccumulator:
        """The raw data-only accumulator, for persistence."""
        return {
            "a": [row[:] for row in self._acc["a"]],
            "g": list(self._acc["g"]),
        }
