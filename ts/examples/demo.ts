/**
 * Demo: token-calibrator in action.
 *
 * Showcases prior estimates, learning from observations, and how the model
 * adapts its coefficients.
 *
 * Run: npx tsx examples/demo.ts
 */

import {
  TokenCalibrator,
  estimateTokensFromPriors,
} from '../src/calibrator.js';

// 1. Stateless prior estimate
const english = "Hello, world! This is a test of the token calibrator.";
const chinese = "你好世界，这是一个测试。";
const mixed   = "Hello 你好 123 🎉";

console.log("=== token-calibrator demo (TypeScript) ===\n");
console.log("── Priors (no learning yet) ──");
console.log(`English  : ${english.length.toString().padStart(4)} chars → ~${estimateTokensFromPriors(english).toString().padStart(3)} tokens`);
console.log(`Chinese  : ${[...chinese].length.toString().padStart(4)} chars → ~${estimateTokensFromPriors(chinese).toString().padStart(3)} tokens`);
console.log(`Mixed    : ${[...mixed].length.toString().padStart(4)} chars → ~${estimateTokensFromPriors(mixed).toString().padStart(3)} tokens`);

// 2. Create a calibrator and learn
const cal = new TokenCalibrator();
console.log("\n── Before observing ──");
console.log(`English estimate: ${cal.estimate(english)} tokens`);
console.log(`Coefficients    :`, cal.coefficients());

// 3. Feed two observations
cal.observe("Hello world", 3);
cal.observe("你好世界", 6);

console.log("\n── After observing 2 rounds ──");
console.log(`English estimate: ${cal.estimate(english)} tokens (was ${estimateTokensFromPriors(english)})`);
console.log(`Coefficients    :`, cal.coefficients());

// 4. Feed more data
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
console.log(`\n── After ${moreData.length} more observations ──`);
console.log(`English estimate: ${cal.estimate(english)} tokens`);
console.log(`Chinese estimate: ${cal.estimate(chinese)} tokens`);
console.log(`Coefficients    :`, cal.coefficients());

// 5. Snapshot round-trip
const snap = cal.snapshot();
const restored = new TokenCalibrator({}, snap);
console.log(`\n── Restored from snapshot ──`);
console.log(`English estimate: ${restored.estimate(english)} (match = ${restored.estimate(english) === cal.estimate(english)})`);

console.log("\n=== Demo complete ===");
