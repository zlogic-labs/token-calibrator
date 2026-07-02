"""
Calibration — the pure ridge-regression math that turns observed
(text, realTokens) rounds into per-bucket rates. Model-blind and stateless;
the stateful writer/reader wrappers live in `calibrator.py` / `estimator.py`.

Model: realTokens ≈ Σ_bucket rate_bucket · count_bucket, fit online by
ridge-regularized least squares. The accumulator stores DATA ONLY; the
ridge prior (λI + λ·prior) is injected only at solve time
(`rates_from_accumulator`). Consequences of keeping the prior out of storage:
- a persisted matrix is a plain data sum;
- forgetting decays only data, so the prior can never erode (no
  strip/re-inject dance — it's structural);
- the prior / its strength can be retuned later without rebaking.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, TypedDict

from .buckets import (
    TOKEN_BUCKETS,
    N_BUCKETS,
    TokenRates,
    TOKEN_BUCKET_PRIORS,
    classify_token_buckets,
    feature_vector,
)


class TokenAccumulator(TypedDict):
    """Data-only least-squares accumulator: a = Σ xxT (N×N), g = Σ x·y (N).
    The ridge prior is NOT baked in — it's added in `rates_from_accumulator`."""

    a: List[List[float]]
    g: List[float]


class AccumulateOptions(TypedDict, total=False):
    forgetting: float


class RatesOptions(TypedDict, total=False):
    priorStrength: float
    prior: TokenRates


DEFAULT_PRIOR_STRENGTH = 1_000_000.0


def empty_accumulator() -> TokenAccumulator:
    """A fresh, empty data-only accumulator."""
    return {
        "a": [[0.0] * N_BUCKETS for _ in range(N_BUCKETS)],
        "g": [0.0] * N_BUCKETS,
    }


def accumulate(
    acc: TokenAccumulator,
    input_str: str,
    real_tokens: float,
    opts: Optional[AccumulateOptions] = None,
) -> TokenAccumulator:
    """Fold one observed round into the accumulator and return a NEW one
    (pure — the input is not mutated). `input` is the text whose real token
    count is known; `real_tokens` the provider's count for it. All-zero
    features or non-positive tokens are ignored (returned unchanged) so
    stray empty / metadata-only rounds can't erode the fit."""
    if not (real_tokens > 0):
        return acc
    x = feature_vector(classify_token_buckets(input_str))
    if all(v == 0.0 for v in x):
        return acc

    if opts is not None and opts.get("forgetting") is not None and opts["forgetting"] > 0:
        gamma = min(1.0, opts["forgetting"])
    else:
        gamma = 1.0

    a = copy.deepcopy(acc["a"])
    g = list(acc["g"])

    for i in range(N_BUCKETS):
        if gamma < 1.0:
            g[i] *= gamma
            for j in range(N_BUCKETS):
                a[i][j] *= gamma
        g[i] += x[i] * real_tokens
        for j in range(N_BUCKETS):
            a[i][j] += x[i] * x[j]

    return {"a": a, "g": g}


def rates_from_accumulator(
    acc: TokenAccumulator,
    opts: Optional[RatesOptions] = None,
) -> TokenRates:
    """Solve the ridge-regularized least squares for the per-bucket rates:
    rates = solve(A + λI, g + λ·prior). Injecting the prior HERE (not in
    storage) keeps accumulators data-only and the regularization constant."""
    if opts is not None and opts.get("priorStrength") is not None and opts["priorStrength"] > 0:
        lam = opts["priorStrength"]
    else:
        lam = DEFAULT_PRIOR_STRENGTH
    prior = opts.get("prior", TOKEN_BUCKET_PRIORS) if opts else TOKEN_BUCKET_PRIORS

    a = copy.deepcopy(acc["a"])
    g = list(acc["g"])

    for j, bucket in enumerate(TOKEN_BUCKETS):
        a[j][j] += lam
        g[j] += lam * prior[bucket]

    theta = solve_linear_system(a, g)
    rates: TokenRates = {}
    for j, bucket in enumerate(TOKEN_BUCKETS):
        rates[bucket] = theta[j]
    return rates


def derive_rates(
    matrices: Dict[str, TokenAccumulator],
    opts: Optional[RatesOptions] = None,
) -> Dict[str, TokenRates]:
    """Derive readable per-model rates from a `{model: matrix}` map (e.g.
    for a lightweight display snapshot or debugging). Invalid entries are
    dropped."""
    out: Dict[str, TokenRates] = {}
    for model, acc in matrices.items():
        if is_valid_accumulator(acc):
            out[model] = rates_from_accumulator(acc, opts)
    return out


def _is_finite_number(v: object) -> bool:
    import math
    return isinstance(v, (int, float)) and math.isfinite(v)


def is_valid_accumulator(acc: object) -> bool:
    """Structurally validate an accumulator before trusting it: N×N finite `a`,
    length-N finite `g`. A malformed blob (corruption, schema drift) is
    rejected so callers can fall back to priors instead of emitting NaN."""
    if not isinstance(acc, dict):
        return False
    a = acc.get("a")
    g = acc.get("g")
    if not isinstance(a, list) or len(a) != N_BUCKETS:
        return False
    for row in a:
        if not isinstance(row, list) or len(row) != N_BUCKETS or not all(_is_finite_number(v) for v in row):
            return False
    if not isinstance(g, list) or len(g) != N_BUCKETS or not all(_is_finite_number(v) for v in g):
        return False
    return True


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """Gauss-Jordan elimination with partial pivoting for a small dense system.
    With the ridge term added the matrix is positive-definite, so this is
    stable; a singular column is skipped (its unknown stays 0)."""
    n = len(b)
    m: List[List[float]] = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(m[r][col]) > abs(m[pivot][col]):
                pivot = r
        if abs(m[pivot][col]) < 1e-12:
            continue

        m[col], m[pivot] = m[pivot], m[col]
        diag = m[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / diag
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]

    x = [0.0] * n
    for i in range(n):
        diag = m[i][i]
        x[i] = 0.0 if abs(diag) < 1e-12 else m[i][n] / diag
    return x
