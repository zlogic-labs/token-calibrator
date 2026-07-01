/**
 * Demo: token-calibrator in action — loading from models.json.
 *
 * Shows how to initialise from a model snapshot file, then learn and
 * export your own snapshot.
 *
 * Run: npx tsx examples/demo.ts
 */

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import {
  TokenCalibrator,
  estimateTokensFromPriors,
} from '../src/calibrator.js';

// Resolve path to models/models.json relative to this file
const __dirname = dirname(fileURLToPath(import.meta.url));
const modelsPath = join(__dirname, '../../models/models.json');
const modelsJson = readFileSync(modelsPath, 'utf-8');

const english = "Hello, world! This is a test of the token calibrator.";
const chinese = "你好世界，这是一个测试。";
const mixed   = "Hello 你好 123 🎉";

console.log("=== token-calibrator demo (TypeScript) ===\n");

// 1. Load from models.json
console.log("── Load from models.json ──");
const cal = TokenCalibrator.fromModel('gpt-4o', modelsJson)
  ?? TokenCalibrator.fromModel('default', modelsJson)!;
console.log("Loaded model: gpt-4o");
console.log(`English estimate: ${cal.estimate(english)} tokens`);
console.log(`Coefficients    :`, cal.coefficients());

// 2. Compare with stateless priors
console.log("\n── Stateless priors (no calibration) ──");
console.log(`English  : ${english.length.toString().padStart(4)} chars → ~${estimateTokensFromPriors(english).toString().padStart(3)} tokens`);
console.log(`Chinese  : ${[...chinese].length.toString().padStart(4)} chars → ~${estimateTokensFromPriors(chinese).toString().padStart(3)} tokens`);
console.log(`Mixed    : ${[...mixed].length.toString().padStart(4)} chars → ~${estimateTokensFromPriors(mixed).toString().padStart(3)} tokens`);

// 3. Feed observations
console.log("\n── Training with real observations ──");
cal.observe("Hello world", 3);
cal.observe("你好世界", 6);

const moreData: [string, number][] = [
  ["short", 2],
  ["a bit longer english text here", 8],
  ["more english words for the model to learn from", 12],
  ["中文中文中文中文", 12],
  ["1234567890", 4],
];
for (const [text, tokens] of moreData) {
  cal.observe(text, tokens);
}
console.log(`After ${2 + moreData.length} observations:`);
console.log(`English estimate: ${cal.estimate(english)} tokens`);
console.log(`Chinese estimate: ${cal.estimate(chinese)} tokens`);
console.log(`Coefficients    :`, cal.coefficients());

// 4. Snapshot round-trip — load fresh from model and replay training
console.log("\n── Restored from fresh model + same training ──");
const restored = TokenCalibrator.fromModel('gpt-4o', modelsJson)
  ?? TokenCalibrator.fromModel('default', modelsJson)!;
restored.observe("Hello world", 3);
restored.observe("你好世界", 6);
for (const [text, tokens] of moreData) {
  restored.observe(text, tokens);
}
console.log(`English estimate: ${restored.estimate(english)} (match = ${restored.estimate(english) === cal.estimate(english)})`);

// 5. Export your trained snapshot (to contribute back!)
console.log("\n── Your trained snapshot (ready to contribute!) ──");
const trained = cal.snapshot();
console.log("a:", JSON.stringify(trained.a));
console.log("g:", JSON.stringify(trained.g));
console.log("strength:", trained.strength);

console.log("\n=== Demo complete ===");
