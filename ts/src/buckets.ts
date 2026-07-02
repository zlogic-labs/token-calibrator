/**
 * Bucketing + pure estimation — the model-blind feature layer.
 *
 * A text is classified into per-bucket counts (`classifyTokenBuckets`), then
 * turned into a token estimate given a set of per-bucket rates
 * (`estimateTokens`). Both are pure and stateless; the fitting of `rates`
 * lives in `calibration.ts`, the stateful wrappers in `calibrator.ts` /
 * `estimator.ts`.
 *
 * Buckets: Han / Latin / Digit / Hangul / Cyrillic / Emoji / Other. Every
 * bucket is a CHARACTER count except `other`, which is a UTF-8 BYTE count —
 * scripts outside a tokenizer's merge vocab fall back to byte-level BPE
 * (≈ 1 token/byte), so byte count tracks their cost better than char count.
 */
export const TOKEN_BUCKETS = [
  'han',
  'latin',
  'digit',
  'hangul',
  'cyrillic',
  'emoji',
  'other',
] as const;

export type TokenBucket = (typeof TOKEN_BUCKETS)[number];

/** Number of buckets — the dimension of the accumulator / feature vector. */
export const N_BUCKETS = TOKEN_BUCKETS.length;

/** Per-bucket token rates. All are per CHARACTER except `other`, which is per
 * UTF-8 BYTE (the byte-level fallback bucket for un-bucketed scripts).
 */
export type TokenRates = Record<TokenBucket, number>;

/** Default per-bucket priors (the ridge target when a model has no / little
 * data). Also what `estimateTokens(input, TOKEN_BUCKET_PRIORS)` yields.
 * Scripts that tokenize very differently (Hangul, Cyrillic, emoji) get their
 * own buckets so one shared `other` rate can't smear them together.
 */
export const TOKEN_BUCKET_PRIORS: TokenRates = {
  han: 1.0,
  latin: 0.25,
  digit: 0.4,
  hangul: 1.2,
  cyrillic: 0.5,
  emoji: 1.5,
  other: 0.6,
};

function utf8Len(cp: number): number {
  if (cp <= 0x7f) return 1;
  if (cp <= 0x7ff) return 2;
  if (cp <= 0xffff) return 3;
  return 4;
}

/**
 * Single-pass classification into the buckets. Every bucket is a CHARACTER
 * count except `other`, which is a UTF-8 BYTE count (byte-level fallback for
 * scripts we don't bucket explicitly).
 */
export function classifyTokenBuckets(input: string): Record<TokenBucket, number> {
  const counts: Record<TokenBucket, number> = {
    han: 0,
    latin: 0,
    digit: 0,
    hangul: 0,
    cyrillic: 0,
    emoji: 0,
    other: 0,
  };
  if (!input) return counts;

  for (const ch of input) {
    const cp = ch.codePointAt(0) ?? 0;

    if (cp >= 0x30 && cp <= 0x39) {
      counts.digit += 1;
    } else if (
        // CJK ideographs across the BMP AND the Supplementary Ideographic
        // Plane, so rare / classical / name characters land in `han` rather
        // than falling through to the byte-counted `other` bucket.
        (cp >= 0x3400 && cp <= 0x4dbf) || // Extension A
        (cp >= 0x4e00 && cp <= 0x9fff) || // Unified (URO)
        (cp >= 0xf900 && cp <= 0xfaff) || // Compatibility Ideographs
        (cp >= 0x20000 && cp <= 0x2a6df) || // Extension B
        (cp >= 0x2a700 && cp <= 0x2ebef) || // Extensions C–F (contiguous)
        (cp >= 0x2f800 && cp <= 0x2fa1f) || // Compatibility Supplement
        (cp >= 0x30000 && cp <= 0x323af) // Extensions G–H
    ) {
      counts.han += 1;
    } else if (cp >= 0x20 && cp <= 0x024f) {
      // Printable ASCII + Latin-1 Supplement + Latin Extended-A: letters,
      // punctuation and spaces. The ~4-chars-per-token prior is calibrated
      // on English text INCLUDING its spaces/punctuation, so folding those
      // here keeps the prior meaningful. Control chars (< 0x20) are excluded
      // — a lone `\n`/`\t` tokenizes unlike a word and belongs in `other`.
      counts.latin += 1;
    } else if (
        (cp >= 0xac00 && cp <= 0xd7a3) || // Hangul Syllables
        (cp >= 0x1100 && cp <= 0x11ff) || // Hangul Jamo
        (cp >= 0x3130 && cp <= 0x318f) // Hangul Compatibility Jamo
    ) {
      counts.hangul += 1;
    } else if (
        (cp >= 0x0400 && cp <= 0x04ff) || // Cyrillic
        (cp >= 0x0500 && cp <= 0x052f) // Cyrillic Supplement
    ) {
      counts.cyrillic += 1;
    } else if (
        (cp >= 0x1f000 && cp <= 0x1faff) || // pictographs, emoticons, transport, flags
        (cp >= 0x2600 && cp <= 0x26ff) || // Miscellaneous Symbols
        (cp >= 0x2700 && cp <= 0x27bf) // Dingbats
    ) {
      counts.emoji += 1;
    } else {
      // Control chars, kana, Arabic, Thai, ZWJ / variation selectors, and
      // every other un-bucketed script → byte-level fallback territory.
      // Count UTF-8 bytes, not characters.
      counts.other += utf8Len(cp);
    }
  }

  return counts;
}

/**
 * Ordered feature vector for the regression: bucket counts in
 * `TOKEN_BUCKETS` order.
 */
export function featureVector(counts: Record<TokenBucket, number>): number[] {
  return TOKEN_BUCKETS.map((b) => counts[b]);
}

/**
 * Pure estimate: input + per-bucket rates → token count. Model-blind.
 */
export function estimateTokens(input: string, rates: TokenRates): number {
  if (!input) return 0;
  const counts = classifyTokenBuckets(input);
  let sum = 0;
  // Floor rates at 0: a pathological fit must never make more text estimate
  // to fewer tokens.
  for (const bucket of TOKEN_BUCKETS) {
    sum += Math.max(0, rates[bucket]) * counts[bucket];
  }
  return Math.max(1, Math.round(sum));
}