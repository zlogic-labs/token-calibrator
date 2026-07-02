import { describe, it, expect } from 'vitest';
import { TokenCalibrator } from './calibrator.js';
import { TokenEstimator } from './estimator.js';
import { classifyTokenBuckets, estimateTokens, TOKEN_BUCKETS, TOKEN_BUCKET_PRIORS, featureVector } from './buckets.js';
import { emptyAccumulator, accumulate, ratesFromAccumulator, solveLinearSystem, isValidAccumulator } from './calibration.js';
import { BUILTIN_TOKEN_RATES } from './builtin-rates.js';

// ---------------------------------------------------------------------------
// classifyTokenBuckets
// ---------------------------------------------------------------------------

describe('classifyTokenBuckets', () => {
  it('classifies Latin text', () => {
    const c = classifyTokenBuckets('Hello');
    expect(c.han).toBe(0);
    expect(c.latin).toBe(5);
    expect(c.digit).toBe(0);
    expect(c.hangul).toBe(0);
    expect(c.cyrillic).toBe(0);
    expect(c.emoji).toBe(0);
    expect(c.other).toBe(0);
  });

  it('classifies Chinese text', () => {
    const c = classifyTokenBuckets('你好');
    expect(c.han).toBe(2);
    expect(c.latin).toBe(0);
    expect(c.other).toBe(0);
  });

  it('classifies mixed text', () => {
    const c = classifyTokenBuckets('Hello 世界 123 😊');
    expect(c.han).toBe(2);
    expect(c.latin).toBe(8);
    expect(c.digit).toBe(3);
    expect(c.emoji).toBe(1);
  });

  it('returns zeros for empty string', () => {
    const c = classifyTokenBuckets('');
    for (const b of TOKEN_BUCKETS) {
      expect(c[b]).toBe(0);
    }
  });

  it('classifies Korean text', () => {
    const c = classifyTokenBuckets('안녕하세요');
    expect(c.hangul).toBe(5);
    expect(c.other).toBe(0);
  });

  it('classifies Cyrillic text', () => {
    const c = classifyTokenBuckets('Привет');
    expect(c.cyrillic).toBe(6);
    expect(c.other).toBe(0);
  });

  it('classifies emoji text', () => {
    const c = classifyTokenBuckets('🚀🎉');
    expect(c.emoji).toBe(2);
    expect(c.other).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// featureVector
// ---------------------------------------------------------------------------

describe('featureVector', () => {
  it('returns ordered vector', () => {
    const fv = featureVector({ han: 2, latin: 5, digit: 3, hangul: 0, cyrillic: 0, emoji: 1, other: 4 });
    expect(fv).toEqual([2, 5, 3, 0, 0, 1, 4]);
  });
});

// ---------------------------------------------------------------------------
// estimateTokens
// ---------------------------------------------------------------------------

describe('estimateTokens', () => {
  it('estimates "Hello world" as 3 tokens', () => {
    expect(estimateTokens('Hello world', TOKEN_BUCKET_PRIORS)).toBe(3);
  });

  it('returns 0 for empty string', () => {
    expect(estimateTokens('', TOKEN_BUCKET_PRIORS)).toBe(0);
  });

  it('floors at 1', () => {
    expect(estimateTokens('a', TOKEN_BUCKET_PRIORS)).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// emptyAccumulator / isValidAccumulator
// ---------------------------------------------------------------------------

describe('isValidAccumulator', () => {
  it('empty is valid', () => {
    expect(isValidAccumulator(emptyAccumulator())).toBe(true);
  });

  it('null is invalid', () => {
    expect(isValidAccumulator(null)).toBe(false);
  });

  it('NaN is invalid', () => {
    const acc = emptyAccumulator();
    acc.a[0][0] = NaN;
    expect(isValidAccumulator(acc)).toBe(false);
  });

  it('Infinity is invalid', () => {
    const acc = emptyAccumulator();
    acc.g[0] = Infinity;
    expect(isValidAccumulator(acc)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// accumulate / ratesFromAccumulator
// ---------------------------------------------------------------------------

describe('accumulate', () => {
  it('ignores non-positive tokens', () => {
    const acc = emptyAccumulator();
    const acc2 = accumulate(acc, 'Hello', 0);
    expect(acc2).toBe(acc);
  });

  it('ignores zero features', () => {
    const acc = emptyAccumulator();
    const acc2 = accumulate(acc, '', 10);
    expect(acc2).toBe(acc);
  });

  it('converges to data-driven rate', () => {
    let acc = emptyAccumulator();
    for (let i = 0; i < 20; i++) {
      acc = accumulate(acc, 'Hello world', 4);
    }
    const rates = ratesFromAccumulator(acc, { priorStrength: 1 });
    expect(rates.latin).toBeCloseTo(4.0 / 11.0, 1);
  });
});

// ---------------------------------------------------------------------------
// solveLinearSystem
// ---------------------------------------------------------------------------

describe('solveLinearSystem', () => {
  it('solves identity system', () => {
    const a = [
      [1, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 1],
    ];
    const b = [2, 3, 4, 5, 6, 7, 8];
    const x = solveLinearSystem(a, b);
    expect(x).toEqual([2, 3, 4, 5, 6, 7, 8]);
  });
});

// ---------------------------------------------------------------------------
// TokenCalibrator
// ---------------------------------------------------------------------------

describe('TokenCalibrator', () => {
  it('starts with prior rates', () => {
    const cal = new TokenCalibrator({ priorStrength: 1 });
    const r = cal.rates();
    expect(r.han).toBeCloseTo(1.0);
    expect(r.latin).toBeCloseTo(0.25);
    expect(r.digit).toBeCloseTo(0.4);
    expect(r.other).toBeCloseTo(0.6);
  });

  it('adapts after observations', () => {
    const cal = new TokenCalibrator({ priorStrength: 1 });
    for (let i = 0; i < 20; i++) {
      cal.observe('Hello world', 4);
    }
    const r = cal.rates();
    expect(r.latin).toBeCloseTo(4.0 / 11.0, 1);
  });

  it('estimates from prior', () => {
    const cal = new TokenCalibrator({ priorStrength: 1 });
    expect(cal.estimate('Hello world')).toBe(3);
  });

  it('estimates after training', () => {
    const cal = new TokenCalibrator({ priorStrength: 1 });
    for (let i = 0; i < 20; i++) {
      cal.observe('Hello world', 4);
    }
    expect(cal.estimate('Hello world')).toBe(4);
  });

  it('round-trips via matrix', () => {
    const cal = new TokenCalibrator();
    cal.observe('test', 2);
    const matrix = cal.toMatrix();
    const restored = new TokenCalibrator({ accumulator: matrix });
    expect(restored.estimate('test')).toBe(cal.estimate('test'));
    expect(restored.rates()).toEqual(cal.rates());
  });

  it('ignores corrupt matrix', () => {
    const cal = new TokenCalibrator({ accumulator: { a: [[1]], g: [1] } as any });
    expect(isValidAccumulator(cal.toMatrix())).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// TokenEstimator
// ---------------------------------------------------------------------------

describe('TokenEstimator', () => {
  it('falls back to baseline for known model', () => {
    const est = new TokenEstimator();
    if ('gpt-4o' in BUILTIN_TOKEN_RATES) {
      const t = est.estimate('gpt-4o', 'Hello world');
      expect(t).toBeGreaterThan(0);
    }
  });

  it('estimates using registered model', () => {
    let acc = emptyAccumulator();
    for (let i = 0; i < 20; i++) {
      acc = accumulate(acc, 'Hello world', 4);
    }
    const est = new TokenEstimator({ 'test-model': acc }, { priorStrength: 1 });
    expect(est.estimate('test-model', 'Hello world')).toBe(4);
  });

  it('returns 0 for empty input', () => {
    const est = new TokenEstimator({ m: emptyAccumulator() });
    expect(est.estimate('m', '')).toBe(0);
  });

  it('checks model existence', () => {
    const est = new TokenEstimator({ m: emptyAccumulator() });
    expect(est.has('m')).toBe(true);
    expect(est.has('nonexistent')).toBe(false);
  });

  it('falls back to prior for unknown model', () => {
    const est = new TokenEstimator();
    const r = est.rates('nonexistent-model-xyz');
    expect(r).toEqual(TOKEN_BUCKET_PRIORS);
  });
});
