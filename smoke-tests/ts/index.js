/**
 * Smoke test: @zlogic/token-calibrator (npm package)
 * Run: npm install && node index.js
 */
import { TokenCalibrator, TokenEstimator, classifyTokenBuckets, estimateTokens, TOKEN_BUCKET_PRIORS, BUILTIN_TOKEN_RATES } from '@zlogic/token-calibrator';

// 1. Basic classify
const c = classifyTokenBuckets('Hello 世界 123');
console.log('[TS] classifyTokenBuckets:', JSON.stringify(c));

// 2. Prior estimate
const t1 = estimateTokens('Hello world', TOKEN_BUCKET_PRIORS);
console.log('[TS] estimate (prior):', t1, '(expected 3)');

// 3. Calibrate + estimate
const cal = new TokenCalibrator({ priorStrength: 1_000 });
const observations = [
  ['Hello world', 3],
  ['你好世界', 6],
  ['12345', 5],
];
for (const [text, tokens] of observations) {
  cal.observe(text, tokens);
}
const t2 = cal.estimate('Hello world');
console.log('[TS] calibrate + estimate:', t2, '(expected ~3)');

// 4. Estimator with built-in models
const est = new TokenEstimator();
const modelNames = Object.keys(BUILTIN_TOKEN_RATES).slice(0, 3);
for (const model of modelNames) {
  const t = est.estimate(model, 'Hello world');
  console.log(`[TS] ${model}:`, t, 'tokens');
}

// 5. Estimator with custom matrix
const matrix = cal.toMatrix();
const est2 = new TokenEstimator({ 'my-model': matrix });
console.log('[TS] custom model estimate:', est2.estimate('my-model', 'Hello world'));

console.log('[TS] All smoke tests passed!');
