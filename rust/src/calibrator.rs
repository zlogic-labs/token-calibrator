//! TokenCalibrator (writer) — the stateful producer side. Wraps ONE model's
//! data-only accumulator: feed it real usage via `observe`, read back solved
//! `rates()` or the raw `to_matrix()` for persistence.

use crate::buckets::{estimate_tokens, TokenRates};
use crate::calibration::{
    accumulate, empty_accumulator, is_valid_accumulator, rates_from_accumulator, TokenAccumulator,
};

/// Options for constructing a `TokenCalibrator`.
#[derive(Clone, Default)]
pub struct TokenCalibratorOptions {
    pub forgetting: Option<f64>,
    pub prior_strength: Option<f64>,
    pub prior: Option<TokenRates>,
    pub accumulator: Option<TokenAccumulator>,
}

/// Stateful writer around ONE model's accumulator.
pub struct TokenCalibrator {
    acc: TokenAccumulator,
    forgetting: Option<f64>,
    prior_strength: Option<f64>,
    prior: Option<TokenRates>,
}

impl TokenCalibrator {
    /// Create a new `TokenCalibrator`.
    pub fn new(opts: TokenCalibratorOptions) -> Self {
        let acc = match opts.accumulator {
            Some(ref a) if is_valid_accumulator(a) => a.clone(),
            _ => empty_accumulator(),
        };
        TokenCalibrator {
            acc,
            forgetting: opts.forgetting,
            prior_strength: opts.prior_strength,
            prior: opts.prior,
        }
    }

    /// Fold in one observed round.
    pub fn observe(&mut self, input: &str, real_tokens: f64) {
        self.acc = accumulate(&self.acc, input, real_tokens, self.forgetting);
    }

    /// Current solved per-bucket rates.
    pub fn rates(&self) -> TokenRates {
        rates_from_accumulator(&self.acc, self.prior_strength, self.prior.as_ref())
    }

    /// Convenience: estimate directly from this calibrator's current rates.
    pub fn estimate(&self, input: &str) -> u32 {
        estimate_tokens(input, &self.rates())
    }

    /// The raw data-only accumulator, for persistence.
    pub fn to_matrix(&self) -> TokenAccumulator {
        self.acc.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starts_with_prior_rates() {
        let cal = TokenCalibrator::new(TokenCalibratorOptions {
            prior_strength: Some(1.0),
            ..Default::default()
        });
        let r = cal.rates();
        assert!((*r.get("han").unwrap() - 1.0).abs() < 1e-9);
        assert!((*r.get("latin").unwrap() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_adapts_after_observations() {
        let mut cal = TokenCalibrator::new(TokenCalibratorOptions {
            prior_strength: Some(1.0),
            ..Default::default()
        });
        for _ in 0..20 {
            cal.observe("Hello world", 4.0);
        }
        let r = cal.rates();
        let latin = *r.get("latin").unwrap();
        assert!((latin - 4.0 / 11.0).abs() < 0.1);
    }

    #[test]
    fn test_estimate_prior() {
        let cal = TokenCalibrator::new(TokenCalibratorOptions {
            prior_strength: Some(1.0),
            ..Default::default()
        });
        assert_eq!(cal.estimate("Hello world"), 3);
    }

    #[test]
    fn test_estimate_after_training() {
        let mut cal = TokenCalibrator::new(TokenCalibratorOptions {
            prior_strength: Some(1.0),
            ..Default::default()
        });
        for _ in 0..20 {
            cal.observe("Hello world", 4.0);
        }
        assert_eq!(cal.estimate("Hello world"), 4);
    }

    #[test]
    fn test_round_trip_matrix() {
        let mut cal = TokenCalibrator::new(TokenCalibratorOptions::default());
        cal.observe("test", 2.0);
        let matrix = cal.to_matrix();
        let restored = TokenCalibrator::new(TokenCalibratorOptions {
            accumulator: Some(matrix),
            ..Default::default()
        });
        assert_eq!(restored.estimate("test"), cal.estimate("test"));
        assert_eq!(restored.rates(), cal.rates());
    }

    #[test]
    fn test_corrupt_matrix_ignored() {
        let mut bad = empty_accumulator();
        bad.a[0][0] = f64::NAN; // corrupt
        let cal = TokenCalibrator::new(TokenCalibratorOptions {
            accumulator: Some(bad),
            ..Default::default()
        });
        // Falls back to empty
        assert!(is_valid_accumulator(&cal.to_matrix()));
    }
}
