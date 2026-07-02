/**
 * TokenCalibrator (writer) — the stateful producer side. Wraps ONE model's
 * data-only accumulator: feed it real usage via `observe`, read back solved
 * `rates()` or the raw `toMatrix()` for persistence. All the math is the pure
 * `calibration.ts` functions; this is just per-instance state + convenience.
 */
import { estimateTokens, type TokenRates } from './buckets.js';
import {
  accumulate,
  emptyAccumulator,
  isValidAccumulator,
  ratesFromAccumulator,
  type AccumulateOptions,
  type RatesOptions,
  type TokenAccumulator,
} from './calibration.js';

export type TokenCalibratorOptions = RatesOptions & AccumulateOptions & {
  /** Resume from a persisted matrix (e.g. `toMatrix()` from a prior run).
   * A corrupt matrix is ignored and a fresh accumulator is used.
   */
  accumulator?: TokenAccumulator;
};

/**
 * Stateful writer around ONE model's accumulator.
 *
 * const cal = new TokenCalibrator();
 * cal.observe(requestText, realInputTokens);
 * const rates = cal.rates(); // for estimation
 * const matrix = cal.toMatrix(); // { [model]: matrix } → persist
 */
export class TokenCalibrator {
  private acc: TokenAccumulator;
  private readonly forgetting?: number;
  private readonly priorStrength?: number;
  private readonly prior?: TokenRates;

  constructor(opts: TokenCalibratorOptions = {}) {
    this.forgetting = opts.forgetting;
    this.priorStrength = opts.priorStrength;
    this.prior = opts.prior;

    this.acc = opts.accumulator && isValidAccumulator(opts.accumulator)
        ? {
          a: opts.accumulator.a.map((row) => row.slice()),
          g: opts.accumulator.g.slice(),
        }
        : emptyAccumulator();
  }

  /** Fold in one observed round. */
  observe(input: string, realTokens: number): void {
    this.acc = accumulate(this.acc, input, realTokens, {
      forgetting: this.forgetting,
    });
  }

  /** Current solved per-bucket rates. */
  rates(): TokenRates {
    return ratesFromAccumulator(this.acc, {
      priorStrength: this.priorStrength,
      prior: this.prior,
    });
  }

  /** Convenience: estimate directly from this calibrator's current rates.
   * For the per-model path prefer `TokenEstimator`; this is sugar for when
   * you already hold a single calibrator.
   */
  estimate(input: string): number {
    return estimateTokens(input, this.rates());
  }

  /** The raw data-only accumulator, for persistence. Store under a model
   * key: `{ [model]: cal.toMatrix() }`; merge is a plain object spread.
   */
  toMatrix(): TokenAccumulator {
    return {
      a: this.acc.a.map((row) => row.slice()),
      g: this.acc.g.slice(),
    };
  }
}