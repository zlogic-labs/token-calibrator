/**
 * token-calibrator demo
 *
 * Demonstrates two usage modes:
 *   1. Train mode  — feed sample observations and export the accumulator
 *   2. Estimate mode — load trained data (or use built-in models) and estimate
 *
 * Usage:
 *   npx tsx examples/demo.ts train        # train & export to trained-snapshot.json
 *   npx tsx examples/demo.ts estimate     # estimate using built-in models
 *   npx tsx examples/demo.ts estimate trained-snapshot.json  # use custom snapshot
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { TokenCalibrator } from '../src/calibrator.js';
import { TokenEstimator } from '../src/estimator.js';
import { BUILTIN_TOKEN_RATES } from '../src/builtin-rates.js';
import { classifyTokenBuckets, estimateTokens } from '../src/buckets.js';
import type { TokenAccumulator } from '../src/calibration.js';

// ───────────────────────── Train mode ─────────────────────────

function cmdTrain(): void {
  console.log('=== Train mode ===\n');

  // Create a calibrator with a small prior strength so data dominates quickly.
  const cal = new TokenCalibrator({ priorStrength: 1_000 });

  // Simulated real usage observations (prompt text, actual token count from API).
  const observations: [string, number][] = [
    ['Hello world',                       3],
    ['The quick brown fox jumps over the lazy dog',  10],
    ['你好，世界',                         6],
    ['안녕하세요',                         8],
    ['Привет мир',                         5],
    ['123 456 7890',                       6],
    ['🚀 Token estimation is amazing! 🎉', 12],
    ['Mixed 你好 Hello 123 😊',             9],
  ];

  for (const [text, tokens] of observations) {
    const counts = classifyTokenBuckets(text);
    console.log(
      `  observe: ${JSON.stringify(counts)}  →  ${tokens} tokens  (${JSON.stringify(text)})`,
    );
    cal.observe(text, tokens);
  }

  // Show learned rates.
  const rates = cal.rates();
  console.log('\nLearned per-bucket rates:');
  for (const [bucket, rate] of Object.entries(rates)) {
    console.log(`  ${bucket.padEnd(10)} ${rate.toFixed(4)}`);
  }

  // Estimate a few samples.
  console.log('\nEstimates after training:');
  for (const text of ['Hello world', '你好', 'Mixed 你好 123 😊']) {
    console.log(`  ${JSON.stringify(text).padEnd(30)} → ${cal.estimate(text)} tokens`);
  }

  // Export accumulator to file.
  const matrix = cal.toMatrix();
  const snapshot: Record<string, TokenAccumulator> = { 'demo-model': matrix };
  writeFileSync('trained-snapshot.json', JSON.stringify({ models: snapshot }, null, 2));
  console.log('\nExported accumulator to trained-snapshot.json\n');
}

// ───────────────────── Estimate mode ─────────────────────

function cmdEstimate(snapshotPath?: string): void {
  console.log('=== Estimate mode ===\n');

  let matrices: Record<string, TokenAccumulator> = {};

  if (snapshotPath && existsSync(snapshotPath)) {
    // Load custom snapshot.
    const raw = JSON.parse(readFileSync(snapshotPath, 'utf-8'));
    matrices = raw.models ?? raw;
    console.log(`Loaded ${Object.keys(matrices).length} model(s) from ${snapshotPath}\n`);
  } else {
    console.log('Using built-in default models (no snapshot file provided)\n');
  }

  const est = new TokenEstimator(matrices);

  // Test texts covering different scripts.
  const testTexts = [
    'Hello world',
    'The quick brown fox jumps over the lazy dog',
    '你好，世界',
    '안녕하세요',
    'Привет мир',
    '123 456 7890',
    '🚀 Token estimation is amazing! 🎉',
  ];

  // A few model names to test.
  const modelNames = Object.keys(BUILTIN_TOKEN_RATES).slice(0, 5);
  modelNames.push('demo-model'); // may or may not be present

  for (const model of modelNames) {
    console.log(`── ${model} ──`);
    if (est.has(model)) {
      console.log('  (user-calibrated data loaded)');
    }
    for (const text of testTexts) {
      const t = est.estimate(model, text);
      console.log(`  ${JSON.stringify(text).padEnd(50)} → ${t} tokens`);
    }
    console.log();
  }

  // Also show what falls back to bare priors.
  console.log('── unknown-model (falls back to prior) ──');
  for (const text of ['Hello', '你好']) {
    const t = est.estimate('unknown-model', text);
    console.log(`  ${JSON.stringify(text).padEnd(50)} → ${t} tokens`);
  }
}

// ─────────────────────── Main ───────────────────────

const mode = process.argv[2] ?? 'estimate';
const arg = process.argv[3];

console.log(`token-calibrator demo (mode: ${mode})\n`);

if (mode === 'train') {
  cmdTrain();
} else {
  cmdEstimate(arg);
}
