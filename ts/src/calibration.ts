/**
 * Calibration — the pure ridge-regression math that turns observed
 * (text, realTokens) rounds into per-bucket rates. Model-blind and stateless;
 * the stateful writer/reader wrappers live in `calibrator.ts` / `estimator.ts`.
 *
 * Model: realTokens ≈ Σ_bucket rate_bucket · count_bucket, fit online by
 * ridge-regularized least squares. The accumulator stores DATA ONLY; the
 * ridge prior (λI + λ·prior) is injected only at solve time
 * (`ratesFromAccumulator`). Consequences of keeping the prior out of storage:
 * - a persisted matrix is a plain data sum;
 * - forgetting decays only data, so the prior can never erode (no
 *   strip/re-inject dance — it's structural);
 * - the prior / its strength can be retuned later without rebaking.
 */
import {
  TOKEN_BUCKETS,
  TOKEN_BUCKET_PRIORS,
  N_BUCKETS,
  classifyTokenBuckets,
  featureVector,
  type TokenRates,
} from './buckets.js';

/** Data-only least-squares accumulator: A = Σ xxT (N×N), g = Σ x·y (N).
 * The ridge prior is NOT baked in — it's added in `ratesFromAccumulator`.
 */
export type TokenAccumulator = {
  a: number[][];
  g: number[];
};

/** Default ridge strength (char2-units): roughly how much real data it
 * takes to move a bucket's rate off the prior. ≈ one medium (~1k-char)
 * round. Larger = stickier prior.
 */
export const DEFAULT_PRIOR_STRENGTH = 1_000_000;

/** A fresh, empty data-only accumulator. */
export function emptyAccumulator(): TokenAccumulator {
  return {
    a: Array.from({ length: N_BUCKETS }, () => new Array<number>(N_BUCKETS).fill(0)),
    g: new Array<number>(N_BUCKETS).fill(0),
  };
}

export type AccumulateOptions = {
  /** Exponential forgetting in (0,1]. 1 = never forget (plain
   * accumulation); 0.97 ≈ track the last ~30 rounds. Because the prior
   * lives outside the accumulator, forgetting only ever decays DATA — the
   * ridge regularization stays full-strength at solve time.
   */
  forgetting?: number;
};

/** Fold one observed round into the accumulator and return a NEW one
 * (pure — the input is not mutated). `input` is the text whose real token
 * count is known; `realTokens` the provider's count for it. All-zero
 * features or non-positive tokens are ignored (returned unchanged) so
 * stray empty / metadata-only rounds can't erode the fit.
 */
export function accumulate(
    acc: TokenAccumulator,
    input: string,
    realTokens: number,
    opts: AccumulateOptions = {},
): TokenAccumulator {
  if (!(realTokens > 0)) return acc;
  const x = featureVector(classifyTokenBuckets(input));
  if (x.every((v) => v === 0)) return acc;

  const gamma = opts.forgetting != null && opts.forgetting > 0
      ? Math.min(1, opts.forgetting)
      : 1;

  const a = acc.a.map((row) => row.slice());
  const g = acc.g.slice();

  for (let i = 0; i < N_BUCKETS; i++) {
    if (gamma < 1) {
      g[i] *= gamma;
      for (let j = 0; j < N_BUCKETS; j++) a[i][j] *= gamma;
    }
    g[i] += x[i] * realTokens;
    for (let j = 0; j < N_BUCKETS; j++) a[i][j] += x[i] * x[j];
  }

  return { a, g };
}

export type RatesOptions = {
  /** Ridge strength λ. Default `DEFAULT_PRIOR_STRENGTH`. */
  priorStrength?: number;
  /** Ridge target the rates are pulled toward with no/little data.
   * Default `TOKEN_BUCKET_PRIORS`. (A per-model baseline can be passed
   * here without any change to storage.)
   */
  prior?: TokenRates;
};

/** Solve the ridge-regularized least squares for the per-bucket rates:
 * rates = solve(A + λI, g + λ·prior). Injecting the prior HERE (not in
 * storage) keeps accumulators data-only and the regularization constant.
 */
export function ratesFromAccumulator(acc: TokenAccumulator, opts: RatesOptions = {}): TokenRates {
  const lambda = opts.priorStrength && opts.priorStrength > 0
      ? opts.priorStrength
      : DEFAULT_PRIOR_STRENGTH;
  const prior = opts.prior ?? TOKEN_BUCKET_PRIORS;

  const a = acc.a.map((row) => row.slice());
  const g = acc.g.slice();

  TOKEN_BUCKETS.forEach((bucket, j) => {
    a[j][j] += lambda;
    g[j] += lambda * prior[bucket];
  });

  const theta = solveLinearSystem(a, g);
  const rates = {} as TokenRates;
  TOKEN_BUCKETS.forEach((bucket, j) => {
    rates[bucket] = theta[j];
  });
  return rates;
}

/** Derive readable per-model rates from a `{ [model]: matrix }` map (e.g.
 * for a lightweight display snapshot or debugging). Invalid entries are
 * dropped.
 */
export function deriveRates(
    matrices: Record<string, TokenAccumulator>,
    opts: RatesOptions = {},
): Record<string, TokenRates> {
  const out: Record<string, TokenRates> = {};
  for (const [model, acc] of Object.entries(matrices)) {
    if (isValidAccumulator(acc)) out[model] = ratesFromAccumulator(acc, opts);
  }
  return out;
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** Structurally validate an accumulator before trusting it: N×N finite `a`,
 * length-N finite `g`. A malformed blob (corruption, schema drift) is
 * rejected so callers can fall back to priors instead of emitting NaN.
 */
export function isValidAccumulator(acc: unknown): acc is TokenAccumulator {
  if (!acc || typeof acc !== 'object') return false;
  const { a, g } = acc as Partial<TokenAccumulator>;
  if (!Array.isArray(a) || a.length !== N_BUCKETS) return false;
  for (const row of a) {
    if (!Array.isArray(row) || row.length !== N_BUCKETS || !row.every(isFiniteNumber)) return false;
  }
  if (!Array.isArray(g) || g.length !== N_BUCKETS || !g.every(isFiniteNumber)) return false;
  return true;
}

/** Gauss-Jordan elimination with partial pivoting for a small dense system.
 * With the ridge term added the matrix is positive-definite, so this is
 * stable; a singular column is skipped (its unknown stays 0).
 */
export function solveLinearSystem(a: number[][], b: number[]): number[] {
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