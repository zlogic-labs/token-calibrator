import { describe, it, expect } from 'vitest';
import {
  TokenTrainer,
  TokenEstimator,
  classifyTokenBuckets,
  estimateTokensFromPriors,
} from './calibrator.js';

describe('classifyTokenBuckets', () => {
  it('classifies Latin text', () => {
    const c = classifyTokenBuckets('Hello');
    expect(c.han).toBe(0);
    expect(c.latin).toBe(5);
    expect(c.digit).toBe(0);
    expect(c.other).toBe(0);
  });

  it('classifies Chinese text', () => {
    const c = classifyTokenBuckets('你好');
    expect(c.han).toBe(2);
    expect(c.latin).toBe(0);
    expect(c.digit).toBe(0);
    expect(c.other).toBe(0);
  });

  it('classifies mixed text', () => {
    const c = classifyTokenBuckets('Hello 世界 123 😊');
    expect(c.han).toBe(2);
    expect(c.latin).toBe(8);
    expect(c.digit).toBe(3);
    expect(c.other).toBe(4);
  });

  it('returns zeros for empty string', () => {
    const c = classifyTokenBuckets('');
    expect(c).toEqual({ han: 0, latin: 0, digit: 0, other: 0 });
  });
});

describe('estimateTokensFromPriors', () => {
  it('estimates "Hello world" as 3 tokens', () => {
    expect(estimateTokensFromPriors('Hello world')).toBe(3);
  });

  it('returns 0 for empty string', () => {
    expect(estimateTokensFromPriors('')).toBe(0);
  });
});

// ---- TokenTrainer tests ----

describe('TokenTrainer', () => {
  it('starts with prior coefficients', () => {
    const trainer = new TokenTrainer({ priorStrength: 1 });
    const c = trainer.coefficients();
    expect(c.han).toBeCloseTo(1.0);
    expect(c.latin).toBeCloseTo(0.25);
    expect(c.digit).toBeCloseTo(0.4);
    expect(c.other).toBeCloseTo(0.6);
  });

  it('adapts after observations (small prior strength)', () => {
    const trainer = new TokenTrainer({ priorStrength: 1 });
    for (let i = 0; i < 20; i++) {
      trainer.observe('Hello world', 4);
    }
    const c = trainer.coefficients();
    expect(c.latin).toBeCloseTo(4.0 / 11.0, 1);
  });

  it('round-trips via snapshot', () => {
    const trainer = new TokenTrainer();
    trainer.observe('test', 2);
    const snap = trainer.snapshot();
    const restored = new TokenTrainer({}, snap);
    expect(restored.a).toEqual(trainer.a);
    expect(restored.g).toEqual(trainer.g);
    expect(restored.strength).toBe(trainer.strength);
    expect(restored.coefficients()).toEqual(trainer.coefficients());
  });
});

// ---- TokenEstimator tests ----

describe('TokenEstimator', () => {
  it('returns null for unknown model', () => {
    const est = new TokenEstimator();
    expect(est.estimate('any-model', 'hello')).toBeNull();
  });

  it('estimates using a registered model', () => {
    const est = new TokenEstimator();
    est.addModel('test-model', {
      a: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      g: [1.0, 0.5, 0.8, 1.2],
      strength: 1.0,
    });
    expect(est.estimate('test-model', 'Hello world')).toBe(6);
    expect(est.estimate('unknown', 'Hello world')).toBeNull();
  });

  it('returns 0 for empty input', () => {
    const est = new TokenEstimator();
    est.addModel('m', {
      a: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      g: [1.0, 0.5, 0.8, 1.2],
      strength: 1.0,
    });
    expect(est.estimate('m', '')).toBe(0);
  });

  it('supports removeModel', () => {
    const est = new TokenEstimator();
    est.addModel('m', {
      a: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      g: [1.0, 0.25, 0.4, 0.6],
      strength: 1.0,
    });
    expect(est.size).toBe(1);
    est.removeModel('m');
    expect(est.size).toBe(0);
    expect(est.estimate('m', 'hello')).toBeNull();
  });

  it('lists model names', () => {
    const est = new TokenEstimator();
    const snap = {
      a: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      g: [1.0, 0.25, 0.4, 0.6],
      strength: 1.0,
    };
    est.addModel('alpha', snap);
    est.addModel('beta', { ...snap });
    const names = [...est.modelNames()];
    expect(names).toHaveLength(2);
    expect(names).toContain('alpha');
    expect(names).toContain('beta');
  });

  it('creates from JSON at construction time', () => {
    const json = JSON.stringify({
      models: {
        'test-model': {
          a: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
          g: [1,0.5,0.8,1.2],
          strength: 1.0,
        },
      },
    });
    // Pass JSON to constructor — no separate fromJson() call
    const est = new TokenEstimator(json);
    expect(est.size).toBe(1);
    expect(est.estimate('test-model', 'Hello world')).toBe(6);
  });

  it('returns empty for empty models JSON', () => {
    const json = JSON.stringify({ models: {} });
    const est = new TokenEstimator(json);
    expect(est.size).toBe(0);
  });

  it('gracefully handles malformed JSON', () => {
    const est = new TokenEstimator('not json');
    expect(est.size).toBe(0);
  });

  it('uses built-in default when no JSON given', () => {
    const est = new TokenEstimator();
    expect(est.size).toBe(4);
    expect([...est.modelNames()]).toContain('default');
    expect([...est.modelNames()]).toContain('gpt-4o');
    expect(est.estimate('default', 'Hello world')).toBe(3);
  });
});
