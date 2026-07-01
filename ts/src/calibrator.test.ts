import { describe, it, expect } from 'vitest';
import {
  TokenCalibrator,
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
    // H e l l o + 3 spaces = 8 Latin; 世 界 = 2 Han; 1 2 3 = 3 Digit; 😊 = 4 bytes Other
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

describe('TokenCalibrator', () => {
  it('starts with prior estimates', () => {
    const cal = new TokenCalibrator();
    expect(cal.estimate('Hello world')).toBe(3);
  });

  it('adapts after observations (small prior strength)', () => {
    const cal = new TokenCalibrator({ priorStrength: 1 });
    expect(cal.estimate('Hello world')).toBe(3);
    for (let i = 0; i < 20; i++) {
      cal.observe('Hello world', 4);
    }
    expect(cal.estimate('Hello world')).toBe(4);
  });

  it('round-trips via snapshot', () => {
    const cal = new TokenCalibrator();
    cal.observe('test', 2);
    const snap = cal.snapshot();
    const restored = new TokenCalibrator({}, snap);
    expect(restored.estimate('test')).toBe(cal.estimate('test'));
  });

  it('returns coefficients as a record', () => {
    const cal = new TokenCalibrator();
    const coef = cal.coefficients();
    expect(coef).toHaveProperty('han');
    expect(coef).toHaveProperty('latin');
    expect(coef).toHaveProperty('digit');
    expect(coef).toHaveProperty('other');
  });
});
