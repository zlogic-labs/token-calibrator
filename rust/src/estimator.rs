//! TokenEstimator (reader) — the stateful consumer side (renderer / CLI display).
//! Built from a `{model: matrix}` map; derives + caches each model's rates up front.
//! Pure math is in `calibration`; this is per-model lookup plus the shipped-baseline fallback.

use std::collections::HashMap;

use crate::buckets::{estimate_tokens, token_bucket_priors, TokenRates};
use crate::calibration::{is_valid_accumulator, rates_from_accumulator, TokenAccumulator};
use crate::builtin_rates::BUILTIN_TOKEN_RATES;

/// Options for constructing a `TokenEstimator`.
#[derive(Clone)]
pub struct TokenEstimatorOptions<'a> {
    pub prior_strength: Option<f64>,
    pub baseline: Option<&'a HashMap<String, TokenRates>>,
    pub prior: Option<TokenRates>,
}

impl Default for TokenEstimatorOptions<'_> {
    fn default() -> Self {
        TokenEstimatorOptions {
            prior_strength: None,
            baseline: None,
            prior: None,
        }
    }
}

/// Read-only, per-model estimator. Three-tier fallback per model:
/// user-calibrated rates → shipped baseline → prior.
pub struct TokenEstimator {
    rates_by_model: HashMap<String, TokenRates>,
    baseline: HashMap<String, TokenRates>,
    prior: TokenRates,
}

impl TokenEstimator {
    /// Create a new `TokenEstimator`.
    pub fn new(
        matrices: Option<HashMap<String, TokenAccumulator>>,
        opts: TokenEstimatorOptions,
    ) -> Self {
        let baseline = opts
            .baseline
            .cloned()
            .unwrap_or_else(|| BUILTIN_TOKEN_RATES.clone());
        let prior = opts.prior.unwrap_or_else(token_bucket_priors);

        let mut rates_by_model = HashMap::new();
        if let Some(matrices) = matrices {
            for (model, acc) in matrices {
                if !is_valid_accumulator(&acc) {
                    continue;
                }
                let model_prior = baseline.get(&model).cloned().unwrap_or_else(|| prior.clone());
                let rates = rates_from_accumulator(&acc, opts.prior_strength, Some(&model_prior));
                rates_by_model.insert(model, rates);
            }
        }

        TokenEstimator {
            rates_by_model,
            baseline,
            prior,
        }
    }

    /// Estimated tokens for `input` under `model`'s effective rates.
    pub fn estimate(&self, model: &str, input: &str) -> u32 {
        estimate_tokens(input, &self.rates(model))
    }

    /// Effective rates for `model`: user-calibrated → baseline → prior.
    pub fn rates(&self, model: &str) -> TokenRates {
        self.rates_by_model
            .get(model)
            .cloned()
            .or_else(|| self.baseline.get(model).cloned())
            .unwrap_or_else(|| self.prior.clone())
    }

    /// Whether this model has its OWN calibrated (user-data) rates.
    pub fn has(&self, model: &str) -> bool {
        self.rates_by_model.contains_key(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{accumulate, empty_accumulator};

    #[test]
    fn test_unknown_model_falls_back_to_baseline() {
        let est = TokenEstimator::new(None, TokenEstimatorOptions::default());
        if BUILTIN_TOKEN_RATES.contains_key("gpt-4o") {
            let t = est.estimate("gpt-4o", "Hello world");
            assert!(t > 0);
        }
    }

    #[test]
    fn test_estimate_from_registered_model() {
        let mut acc = empty_accumulator();
        for _ in 0..20 {
            acc = accumulate(&acc, "Hello world", 4.0, None);
        }
        let mut matrices = HashMap::new();
        matrices.insert("test-model".into(), acc);
        let est = TokenEstimator::new(
            Some(matrices),
            TokenEstimatorOptions {
                prior_strength: Some(1.0),
                ..Default::default()
            },
        );
        assert_eq!(est.estimate("test-model", "Hello world"), 4);
    }

    #[test]
    fn test_empty_input() {
        let acc = empty_accumulator();
        let mut matrices = HashMap::new();
        matrices.insert("m".into(), acc);
        let est = TokenEstimator::new(Some(matrices), TokenEstimatorOptions::default());
        assert_eq!(est.estimate("m", ""), 0);
    }

    #[test]
    fn test_has_model() {
        let acc = empty_accumulator();
        let mut matrices = HashMap::new();
        matrices.insert("m".into(), acc);
        let est = TokenEstimator::new(Some(matrices), TokenEstimatorOptions::default());
        assert!(est.has("m"));
        assert!(!est.has("nonexistent"));
    }

    #[test]
    fn test_rates_fallback_tier() {
        let est = TokenEstimator::new(None, TokenEstimatorOptions::default());
        let r = est.rates("nonexistent-model-xyz");
        assert_eq!(r, token_bucket_priors());
    }
}
