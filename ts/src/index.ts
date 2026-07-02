/**
 * Token calibrator — self-learning token estimator for LLM context-size display.
 *
 * Uses online ridge regression to learn per-bucket token rates from real usage
 * observations. No bundled tokenizer required.
 *
 * @packageDocumentation
 */

export { TokenCalibrator, type TokenCalibratorOptions } from './calibrator.js';
export { TokenEstimator, type TokenEstimatorOptions } from './estimator.js';
export {
  TOKEN_BUCKETS,
  N_BUCKETS,
  type TokenBucket,
  type TokenRates,
  TOKEN_BUCKET_PRIORS,
  classifyTokenBuckets,
  featureVector,
  estimateTokens,
} from './buckets.js';
export {
  type TokenAccumulator,
  type AccumulateOptions,
  type RatesOptions,
  DEFAULT_PRIOR_STRENGTH,
  emptyAccumulator,
  accumulate,
  ratesFromAccumulator,
  deriveRates,
  isValidAccumulator,
  solveLinearSystem,
} from './calibration.js';
export { BUILTIN_TOKEN_RATES } from './builtin-rates.js';
