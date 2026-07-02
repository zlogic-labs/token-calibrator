"""
Bucketing + pure estimation — the model-blind feature layer.

A text is classified into per-bucket counts (`classify_token_buckets`), then
turned into a token estimate given a set of per-bucket rates (`estimate_tokens`).
Both are pure and stateless; the fitting of rates lives in `calibration.py`,
the stateful wrappers in `calibrator.py` / `estimator.py`.

Buckets: Han / Latin / Digit / Hangul / Cyrillic / Emoji / Other. Every
bucket is a CHARACTER count except `other`, which is a UTF-8 BYTE count —
scripts outside a tokenizer's merge vocab fall back to byte-level BPE
(≈ 1 token/byte), so byte count tracks their cost better than char count.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

TOKEN_BUCKETS: Tuple[str, ...] = (
    "han",
    "latin",
    "digit",
    "hangul",
    "cyrillic",
    "emoji",
    "other",
)

TokenBucket = str  # one of TOKEN_BUCKETS

N_BUCKETS = len(TOKEN_BUCKETS)

TokenRates = Dict[str, float]

TOKEN_BUCKET_PRIORS: TokenRates = {
    "han": 1.0,
    "latin": 0.25,
    "digit": 0.4,
    "hangul": 1.2,
    "cyrillic": 0.5,
    "emoji": 1.5,
    "other": 0.6,
}


def _utf8_len(cp: int) -> int:
    if cp <= 0x7F:
        return 1
    if cp <= 0x7FF:
        return 2
    if cp <= 0xFFFF:
        return 3
    return 4


def classify_token_buckets(input_str: str) -> Dict[str, int]:
    """
    Single-pass classification into the buckets. Every bucket is a CHARACTER
    count except `other`, which is a UTF-8 BYTE count (byte-level fallback for
    scripts we don't bucket explicitly).
    """
    counts: Dict[str, int] = {
        "han": 0,
        "latin": 0,
        "digit": 0,
        "hangul": 0,
        "cyrillic": 0,
        "emoji": 0,
        "other": 0,
    }
    if not input_str:
        return counts

    for ch in input_str:
        cp = ord(ch)

        if 0x30 <= cp <= 0x39:
            counts["digit"] += 1
        elif (
            (0x3400 <= cp <= 0x4DBF)
            or (0x4E00 <= cp <= 0x9FFF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2EBEF)
            or (0x2F800 <= cp <= 0x2FA1F)
            or (0x30000 <= cp <= 0x323AF)
        ):
            counts["han"] += 1
        elif 0x20 <= cp <= 0x024F:
            counts["latin"] += 1
        elif (
            (0xAC00 <= cp <= 0xD7A3)
            or (0x1100 <= cp <= 0x11FF)
            or (0x3130 <= cp <= 0x318F)
        ):
            counts["hangul"] += 1
        elif (0x0400 <= cp <= 0x04FF) or (0x0500 <= cp <= 0x052F):
            counts["cyrillic"] += 1
        elif (
            (0x1F000 <= cp <= 0x1FAFF)
            or (0x2600 <= cp <= 0x26FF)
            or (0x2700 <= cp <= 0x27BF)
        ):
            counts["emoji"] += 1
        else:
            counts["other"] += _utf8_len(cp)

    return counts


def feature_vector(counts: Dict[str, int]) -> List[float]:
    """Ordered feature vector for the regression: bucket counts in
    `TOKEN_BUCKETS` order."""
    return [float(counts[b]) for b in TOKEN_BUCKETS]


def estimate_tokens(input_str: str, rates: TokenRates) -> int:
    """Pure estimate: input + per-bucket rates → token count. Model-blind."""
    if not input_str:
        return 0
    counts = classify_token_buckets(input_str)
    total = 0.0
    # Floor rates at 0: a pathological fit must never make more text estimate
    # to fewer tokens.
    for bucket in TOKEN_BUCKETS:
        total += max(0.0, rates[bucket]) * counts[bucket]
    return max(1, round(total))
