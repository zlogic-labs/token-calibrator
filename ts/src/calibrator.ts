/**
 * Calibrated, self-learning token estimator for the live context-size
 * display. Kept SEPARATE from `tokens.ts` (the fixed CJK fallback +
 * cost helpers) on purpose — this module owns one job:
 *
 * estimate(input: string) → estimated token count
 *
 * How it stays accurate: it learns per-character token *rates* from the
 * provider's REAL usage counts (`observe`), so estimates converge toward
 * whatever tokenizer the active model actually uses — no bundled
 * tokenizer, no per-vendor config.
 *
 * Model: realTokens ≈ Σ_bucket rate_bucket · count_bucket, fit online by
 * ridge-regularized least squares. The ridge prior is injected as one
 * "pure bucket" pseudo-sample per bucket, so (a) the solve is always
 * well-conditioned (no collinearity blow-up when the history is all one
 * script) and (b) with no real data the coefficients equal the priors
 * exactly, degrading to a plain char-rate heuristic.
 *
 * Buckets (this version): Han / Latin / Digit / Other. Han/Latin/Digit are
 * counted per CHARACTER; Other is counted per UTF-8 BYTE, because a script
 * outside the tokenizer's merge vocab falls back to byte-level BPE
 * (≈ 1 token/byte), so byte count tracks its cost far better than chars.
 */
export const TOKEN_BUCKETS = ['han', 'latin', 'digit', 'other'] as const;
export type TokenBucket = (typeof TOKEN_BUCKETS)[number];

/**
 * Prior token-per-unit rates. han/latin/digit are per CHARACTER; `other`
 * is per UTF-8 BYTE. These seed the calibrator and are exactly what it
 * returns before any real usage has been observed.
 */
export const TOKEN_BUCKET_PRIORS: Record<TokenBucket, number> = {
  han: 1.0,
  latin: 0.25,
  digit: 0.4,
  other: 0.6,
};

const N_BUCKETS = TOKEN_BUCKETS.length;
const DEFAULT_PRIOR_STRENGTH = 1_000_000;

function utf8Len(cp: number): number {
  if (cp <= 0x7f) return 1;
  if (cp <= 0x7ff) return 2;
  if (cp <= 0xffff) return 3;
  return 4;
}

/**
 * Single-pass classification into the four buckets. han/latin/digit are
 * char counts; other is a UTF-8 byte count.
 */
export function classifyTokenBuckets(input: string): Record<TokenBucket, number> {
  const counts: Record<TokenBucket, number> = {
    han: 0,
    latin: 0,
    digit: 0,
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
    } else {
      // Control chars, kana, Hangul, Cyrillic, Arabic, emoji, rare scripts ...
      // → byte-level fallback territory. Count UTF-8 bytes, not characters.
      counts.other += utf8Len(cp);
    }
  }

  return counts;
}

function featureVector(counts: Record<TokenBucket, number>): number[] {
  return TOKEN_BUCKETS.map((b) => counts[b]);
}

/**
 * Stateless estimate straight from the priors — the answer a fresh
 * calibrator gives before it has seen any real usage. Handy where no
 * calibrator instance is threaded through.
 */
export function estimateTokensFromPriors(input: string): number {
  if (!input) return 0;
  const counts = classifyTokenBuckets(input);
  let sum = 0;
  for (const bucket of TOKEN_BUCKETS) {
    sum += TOKEN_BUCKET_PRIORS[bucket] * counts[bucket];
  }
  return Math.max(1, Math.round(sum));
}

/**
 * Gauss-Jordan elimination with partial pivoting for a small dense system.
 * The calibrator's matrix is symmetric positive-definite (ridge seed), so
 * this is stable; a singular column is skipped (its unknown stays 0).
 */
function solveLinearSystem(a: number[][], b: number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    }
    if (Math.abs(m[pivot][col]) < 1e-12) continue;

    [m[col], m[pivot]] = [m[pivot], m[col]];
    const diag = m[col][col];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = m[r][col] / diag;
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) m[r][c] -= factor * m[col][c];
    }
  }

  const x = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    const diag = m[i][i];
    x[i] = Math.abs(diag) < 1e-12 ? 0 : m[i][n] / diag;
  }
  return x;
}

export type TokenCalibratorSnapshot = {
  a: number[][];
  g: number[];
  strength: number;
};

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/**
 * Structurally validate a persisted snapshot before trusting it: right
 * dimensions (4×4 `a`, length-4 `g`), all-finite numbers, positive
 * `strength`. A malformed blob (truncated write, schema drift, corruption)
 * is rejected so the calibrator rebuilds from priors instead of emitting
 * NaN / garbage estimates.
 */
function isValidSnapshot(s: unknown): s is TokenCalibratorSnapshot {
  if (!s || typeof s !== 'object') return false;
  const snap = s as Partial<TokenCalibratorSnapshot>;
  if (!isFiniteNumber(snap.strength) || snap.strength <= 0) return false;
  if (!Array.isArray(snap.a) || snap.a.length !== N_BUCKETS) return false;
  for (const row of snap.a) {
    if (!Array.isArray(row) || row.length !== N_BUCKETS || !row.every(isFiniteNumber)) return false;
  }
  if (!Array.isArray(snap.g) || snap.g.length !== N_BUCKETS || !snap.g.every(isFiniteNumber)) {
    return false;
  }
  return true;
}

export type TokenCalibratorOptions = {
  /**
   * Prior strength in char2-units: how much real data it takes to move a
   * bucket's rate off its prior. Larger = stickier priors. Default ≈ one
   * medium (~1k-char) round.
   */
  priorStrength?: number;
  /**
   * Exponential forgetting in (0,1]. 1 = never forget (plain
   * accumulation); 0.97 ≈ track the last ~30 rounds (handles content
   * drift / a model swap).
   */
  forgetting?: number;
};

/**
 * Online ridge-regression token estimator over the four buckets.
 *
 * Primary interface — meets "input in, estimated tokens out":
 * const est = calibrator.estimate(input);
 *
 * Feed it ground truth as it arrives to sharpen future estimates:
 * calibrator.observe(requestText, realInputTokens);
 */
export class TokenCalibrator {
  private a: number[][];
  private g: number[];
  private readonly gamma: number;
  private readonly strength: number;
  private theta: number[] | null = null;

  constructor(opts: TokenCalibratorOptions = {}, snapshot?: TokenCalibratorSnapshot) {
    this.gamma =
      opts.forgetting != null && opts.forgetting > 0 ? Math.min(1, opts.forgetting) : 1;

    this.a = Array.from({ length: N_BUCKETS }, () => new Array<number>(N_BUCKETS).fill(0));
    this.g = new Array<number>(N_BUCKETS).fill(0);

    if (isValidSnapshot(snapshot)) {
      // The prior is already baked into the persisted a/g; reuse the SAME
      // strength it was seeded with so observe()'s strip/re-inject stays
      // consistent. A corrupt snapshot falls through to a fresh
      // prior-seeded state rather than poisoning estimates.
      this.strength = snapshot.strength;
      this.a = snapshot.a.map((row) => row.slice());
      this.g = snapshot.g.slice();
      return;
    }

    this.strength =
      opts.priorStrength && opts.priorStrength > 0 ? opts.priorStrength : DEFAULT_PRIOR_STRENGTH;

    // One pure-bucket pseudo-sample each → diagonal A, g = strength·prior,
    // so solve() returns the priors exactly until real data arrives.
    this.seedPriors();
  }

  /**
   * Add the ridge prior (constant λ=strength on the diagonal) into a/g.
   */
  private seedPriors(): void {
    TOKEN_BUCKETS.forEach((bucket, j) => {
      this.a[j][j] += this.strength;
      this.g[j] += this.strength * TOKEN_BUCKET_PRIORS[bucket];
    });
  }

  /**
   * Remove the ridge prior so ONLY the accumulated data gets decayed.
   */
  private stripPriors(): void {
    TOKEN_BUCKETS.forEach((bucket, j) => {
      this.a[j][j] -= this.strength;
      this.g[j] -= this.strength * TOKEN_BUCKET_PRIORS[bucket];
    });
  }

  /**
   * Calibrated token estimate for `input`.
   */
  estimate(input: string): number {
    if (!input) return 0;
    const x = featureVector(classifyTokenBuckets(input));
    const theta = this.solve();
    let sum = 0;
    // Floor coefficients at 0: a pathological fit must never make more
    // text estimate to fewer tokens.
    for (let j = 0; j < N_BUCKETS; j++) sum += Math.max(0, theta[j]) * x[j];
    return Math.max(1, Math.round(sum));
  }

  /**
   * Fold in one observed round: `input` whose real token count is known.
   */
  observe(input: string, realTokens: number): void {
    if (!(realTokens > 0)) return;
    const x = featureVector(classifyTokenBuckets(input));

    // All-zero features carry no information; folding them in would only
    // decay history (when forgetting < 1) without adding anything — drop
    // so stray empty / metadata-only usage rounds can't erode the fit.
    if (x.every((v) => v === 0)) return;

    // Peel the prior off so forgetting decays ONLY the accumulated data,
    // then re-inject the SAME constant prior. This keeps the ridge penalty
    // (hence a positive-definite, well-conditioned matrix) intact no matter
    // how many rounds pass — without it, γ<1 would decay the prior toward 0
    // and a run of single-script rounds could then blow the solve up.
    this.stripPriors();

    if (this.gamma < 1) {
      for (let i = 0; i < N_BUCKETS; i++) {
        this.g[i] *= this.gamma;
        for (let j = 0; j < N_BUCKETS; j++) {
          this.a[i][j] *= this.gamma;
        }
      }
    }

    for (let i = 0; i < N_BUCKETS; i++) {
      this.g[i] += x[i] * realTokens;
      for (let j = 0; j < N_BUCKETS; j++) {
        this.a[i][j] += x[i] * x[j];
      }
    }

    this.seedPriors();
    this.theta = null;
  }

  /**
   * Current learned per-bucket rates (for diagnostics / display).
   */
  coefficients(): Record<TokenBucket, number> {
    const theta = this.solve();
    const out = {} as Record<TokenBucket, number>;
    TOKEN_BUCKETS.forEach((bucket, j) => {
      out[bucket] = theta[j];
    });
    return out;
  }

  /**
   * Serializable state — persist to learn across sessions if desired.
   */
  snapshot(): TokenCalibratorSnapshot {
    return {
      a: this.a.map((row) => row.slice()),
      g: this.g.slice(),
      strength: this.strength,
    };
  }

  private solve(): number[] {
    if (!this.theta) this.theta = solveLinearSystem(this.a, this.g);
    return this.theta;
  }
}