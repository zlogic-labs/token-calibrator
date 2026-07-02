/**
 * TokenEstimator (reader) — the stateful consumer side (renderer / CLI
 * display). Built from a `{ [model]: matrix }` map (as produced by
 * calibrators and merged by object spread); derives + caches each model's
 * rates up front. Pure math is in `calibration.ts`; this is per-model lookup
 * plus the shipped-baseline fallback.
 */
import { estimateTokens, TOKEN_BUCKET_PRIORS, type TokenRates } from './buckets.js';
import { isValidAccumulator, ratesFromAccumulator, type TokenAccumulator } from './calibration.js';
import { BUILTIN_TOKEN_RATES } from './builtin-rates.js';

export type TokenEstimatorOptions = {
  /** Ridge strength λ used when deriving rates from user matrices. */
  priorStrength?: number;
  /** Per-model shipped baseline rates. Defaults to `BUILTIN_TOKEN_RATES`.
   * Serves two roles: the ridge prior when deriving a model's user data,
   * AND the estimate for a model that has baseline data but no user data.
   * Pass `{}` to disable the shipped baseline entirely.
   */
  baseline?: Record<string, TokenRates>;
  /** Final fallback when a model has neither user nor baseline data.
   * Defaults to `TOKEN_BUCKET_PRIORS`.
   */
  prior?: TokenRates;
};

/**
 * Read-only, per-model estimator. Three-tier fallback per model:
 *
 * user-calibrated rates → shipped baseline (BUILTIN_TOKEN_RATES) → prior
 *
 * The baseline also acts as the ridge prior when a model DOES have user
 * data, so a lightly-calibrated model stays anchored near its shipped rates.
 *
 * const est = new TokenEstimator(loadedMatrices);
 * const tokens = est.estimate('gpt-5.5', requestBody);
 */
export class TokenEstimator {
  private readonly ratesByModel = new Map<string, TokenRates>();
  private readonly baseline: Record<string, TokenRates>;
  private readonly prior: TokenRates;

  constructor(matrices: Record<string, TokenAccumulator> = {}, opts: TokenEstimatorOptions = {}) {
    this.baseline = opts.baseline ?? BUILTIN_TOKEN_RATES;
    this.prior = opts.prior ?? TOKEN_BUCKET_PRIORS;

    for (const [model, acc] of Object.entries(matrices)) {
      if (!isValidAccumulator(acc)) continue;
      // Anchor each model's fit to its shipped baseline if present, else the
      // generic prior.
      this.ratesByModel.set(
          model,
          ratesFromAccumulator(acc, {
            priorStrength: opts.priorStrength,
            prior: this.baseline[model] ?? this.prior,
          }),
      );
    }
  }

  /** Estimated tokens for `input` under `model`'s effective rates. */
  estimate(model: string, input: string): number {
    return estimateTokens(input, this.rates(model));
  }

  /** Effective rates for `model`: user-calibrated → baseline → prior. */
  rates(model: string): TokenRates {
    return this.ratesByModel.get(model) ?? this.baseline[model] ?? this.prior;
  }

  /** Whether this model has its OWN calibrated (user-data) rates. */
  has(model: string): boolean {
    return this.ratesByModel.has(model);
  }
}