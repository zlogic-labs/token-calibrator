//! Calibration — the pure ridge-regression math that turns observed
//! (text, realTokens) rounds into per-bucket rates. Model-blind and stateless;
//! the stateful writer/reader wrappers live in `calibrator` / `estimator`.
//!
//! Model: realTokens ≈ Σ_bucket rate_bucket · count_bucket, fit online by
//! ridge-regularized least squares. The accumulator stores DATA ONLY; the
//! ridge prior (λI + λ·prior) is injected only at solve time
//! (`rates_from_accumulator`).

use crate::buckets::{classify_token_buckets, feature_vector, token_bucket_priors, TokenRates, N_BUCKETS, TOKEN_BUCKETS};

/// Data-only least-squares accumulator: a = Σ xxT (N×N), g = Σ x·y (N).
#[derive(Clone, Debug, PartialEq)]
pub struct TokenAccumulator {
    pub a: [[f64; N_BUCKETS]; N_BUCKETS],
    pub g: [f64; N_BUCKETS],
}

/// Default ridge strength.
pub const DEFAULT_PRIOR_STRENGTH: f64 = 1_000_000.0;

/// A fresh, empty data-only accumulator.
pub fn empty_accumulator() -> TokenAccumulator {
    TokenAccumulator {
        a: [[0.0; N_BUCKETS]; N_BUCKETS],
        g: [0.0; N_BUCKETS],
    }
}

/// Fold one observed round into the accumulator and return a NEW one (pure).
/// All-zero features or non-positive tokens are ignored.
pub fn accumulate(
    acc: &TokenAccumulator,
    input: &str,
    real_tokens: f64,
    forgetting: Option<f64>,
) -> TokenAccumulator {
    if !(real_tokens > 0.0) {
        return acc.clone();
    }
    let x = feature_vector(&classify_token_buckets(input));
    if x.iter().all(|v| *v == 0.0) {
        return acc.clone();
    }

    let gamma = match forgetting {
        Some(f) if f > 0.0 => f64::min(1.0, f),
        _ => 1.0,
    };

    let mut a = acc.a;
    let mut g = acc.g;

    for i in 0..N_BUCKETS {
        if gamma < 1.0 {
            g[i] *= gamma;
            for j in 0..N_BUCKETS {
                a[i][j] *= gamma;
            }
        }
        g[i] += x[i] * real_tokens;
        for j in 0..N_BUCKETS {
            a[i][j] += x[i] * x[j];
        }
    }

    TokenAccumulator { a, g }
}

/// Solve the ridge-regularized least squares for the per-bucket rates:
/// rates = solve(A + λI, g + λ·prior).
pub fn rates_from_accumulator(
    acc: &TokenAccumulator,
    prior_strength: Option<f64>,
    prior: Option<&TokenRates>,
) -> TokenRates {
    let lambda = prior_strength.filter(|&s| s > 0.0).unwrap_or(DEFAULT_PRIOR_STRENGTH);
    let prior_map = prior.cloned().unwrap_or_else(token_bucket_priors);

    let mut a = acc.a;
    let mut g = acc.g;

    for (j, bucket) in TOKEN_BUCKETS.iter().enumerate() {
        a[j][j] += lambda;
        g[j] += lambda * prior_map.get(*bucket).copied().unwrap_or(0.0);
    }

    let theta = solve_linear_system(&a, &g);
    let mut rates = TokenRates::new();
    for (j, bucket) in TOKEN_BUCKETS.iter().enumerate() {
        rates.insert((*bucket).to_string(), theta[j]);
    }
    rates
}

/// Derive readable per-model rates from a `{model: matrix}` map.
/// Invalid entries are dropped.
pub fn derive_rates(
    matrices: &std::collections::HashMap<String, TokenAccumulator>,
    prior_strength: Option<f64>,
    prior: Option<&TokenRates>,
) -> std::collections::HashMap<String, TokenRates> {
    let mut out = std::collections::HashMap::new();
    for (model, acc) in matrices {
        if is_valid_accumulator(acc) {
            out.insert(model.clone(), rates_from_accumulator(acc, prior_strength, prior));
        }
    }
    out
}

/// Structurally validate an accumulator: N×N finite `a`, length-N finite `g`.
pub fn is_valid_accumulator(acc: &TokenAccumulator) -> bool {
    for row in &acc.a {
        if row.len() != N_BUCKETS {
            return false;
        }
        for &v in row {
            if !v.is_finite() {
                return false;
            }
        }
    }
    if acc.g.len() != N_BUCKETS {
        return false;
    }
    for &v in &acc.g {
        if !v.is_finite() {
            return false;
        }
    }
    true
}

/// Gauss-Jordan elimination with partial pivoting for a small dense system.
fn solve_linear_system(a: &[[f64; N_BUCKETS]; N_BUCKETS], b: &[f64; N_BUCKETS]) -> [f64; N_BUCKETS] {
    let n = N_BUCKETS;
    // augmented matrix [A | b]
    let mut m: Vec<Vec<f64>> = a.iter().enumerate().map(|(i, row)| {
        let mut r: Vec<f64> = row.to_vec();
        r.push(b[i]);
        r
    }).collect();

    for col in 0..n {
        let mut pivot = col;
        for r in (col + 1)..n {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        if m[pivot][col].abs() < 1e-12 {
            continue;
        }

        m.swap(col, pivot);
        let diag = m[col][col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = m[r][col] / diag;
            if factor == 0.0 {
                continue;
            }
            for c in col..=n {
                m[r][c] -= factor * m[col][c];
            }
        }
    }

    let mut x = [0.0f64; N_BUCKETS];
    for i in 0..n {
        let diag = m[i][i];
        x[i] = if diag.abs() < 1e-12 { 0.0 } else { m[i][n] / diag };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_accumulator_is_valid() {
        let acc = empty_accumulator();
        assert!(is_valid_accumulator(&acc));
    }

    #[test]
    fn test_nan_in_a_is_invalid() {
        let mut acc = empty_accumulator();
        acc.a[0][0] = f64::NAN;
        assert!(!is_valid_accumulator(&acc));
    }

    #[test]
    fn test_inf_in_g_is_invalid() {
        let mut acc = empty_accumulator();
        acc.g[0] = f64::INFINITY;
        assert!(!is_valid_accumulator(&acc));
    }

    #[test]
    fn test_non_positive_tokens_ignored() {
        let acc = empty_accumulator();
        let acc2 = accumulate(&acc, "Hello", 0.0, None);
        assert_eq!(acc, acc2);
    }

    #[test]
    fn test_accumulate_and_solve() {
        let mut acc = empty_accumulator();
        for _ in 0..20 {
            acc = accumulate(&acc, "Hello world", 4.0, None);
        }
        let rates = rates_from_accumulator(&acc, Some(1.0), None);
        let latin = *rates.get("latin").unwrap();
        assert!((latin - 4.0 / 11.0).abs() < 0.1);
    }

    #[test]
    fn test_solve_identity() {
        let a = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = solve_linear_system(&a, &b);
        assert_eq!(x, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
}
