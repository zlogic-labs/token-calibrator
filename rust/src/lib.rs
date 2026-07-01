//! Calibrated, self-learning token estimator for the live context-size display.
//!
//! This module provides an online ridge‑regression estimator that learns
//! per‑bucket token rates from real usage observations. It requires no
//! bundled tokenizer and adapts to any model’s actual tokenization behaviour.
//!
//! The four buckets are: Han, Latin, Digit, and Other (counted in UTF‑8 bytes).
//! Priors are supplied and act as a ridge penalty so that the system remains
//! well‑conditioned and degrades gracefully to the heuristic when no data
//! has been observed.

use std::f64;

/// The four classification buckets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenBucket {
    Han,
    Latin,
    Digit,
    Other,
}

/// Number of buckets.
pub const N_BUCKETS: usize = 4;

/// Array type for a bucket‑count vector (used for feature vectors).
pub type BucketCounts = [usize; N_BUCKETS];

/// Array type for a coefficient vector (rates per bucket).
pub type Coefficients = [f64; N_BUCKETS];

/// Matrix type for the Gram matrix (4×4).
pub type Matrix4 = [[f64; N_BUCKETS]; N_BUCKETS];

/// Prior token‑per‑unit rates for each bucket.
/// - Han / Latin / Digit are per character,
/// - Other is per UTF‑8 byte.
pub const TOKEN_BUCKET_PRIORS: Coefficients = [1.0, 0.25, 0.4, 0.6];

/// Default strength of the ridge prior (amount of pseudo‑data).
pub const DEFAULT_PRIOR_STRENGTH: f64 = 1_000_000.0;

/// Names of the buckets (for diagnostics).
pub const TOKEN_BUCKET_NAMES: [&str; N_BUCKETS] = ["han", "latin", "digit", "other"];

// -----------------------------------------------------------------------------
// Auxiliary functions
// -----------------------------------------------------------------------------

/// Returns the number of UTF‑8 bytes needed to encode the given Unicode scalar.
#[inline]
fn utf8_len(cp: u32) -> usize {
    if cp <= 0x7F {
        1
    } else if cp <= 0x7FF {
        2
    } else if cp <= 0xFFFF {
        3
    } else {
        4
    }
}

/// Classify a string into the four buckets.
///
/// - `Han`: CJK ideographs (including extension planes)
/// - `Latin`: printable ASCII, Latin‑1 supplement, Latin Extended‑A
///   (spaces and punctuation are included to keep the prior meaningful)
/// - `Digit`: ASCII digits `0`–`9`
/// - `Other`: everything else, counted in UTF‑8 bytes.
pub fn classify_token_buckets(input: &str) -> BucketCounts {
    let mut counts = [0; N_BUCKETS];
    for ch in input.chars() {
        let cp = ch as u32;
        if (0x30..=0x39).contains(&cp) {
            counts[TokenBucket::Digit as usize] += 1;
        } else if (0x3400..=0x4DBF).contains(&cp)       // Extension A
            || (0x4E00..=0x9FFF).contains(&cp)          // Unified (URO)
            || (0xF900..=0xFAFF).contains(&cp)          // Compatibility Ideographs
            || (0x20000..=0x2A6DF).contains(&cp)        // Extension B
            || (0x2A700..=0x2EBEF).contains(&cp)        // Extensions C–F
            || (0x2F800..=0x2FA1F).contains(&cp)        // Compatibility Supplement
            || (0x30000..=0x323AF).contains(&cp)        // Extensions G–H
        {
            counts[TokenBucket::Han as usize] += 1;
        } else if (0x20..=0x024F).contains(&cp) {
            // Printable ASCII, Latin‑1 Supplement, Latin Extended‑A
            counts[TokenBucket::Latin as usize] += 1;
        } else {
            // Fallback: count bytes
            counts[TokenBucket::Other as usize] += utf8_len(cp);
        }
    }
    counts
}

/// Convert bucket counts to a feature vector (as `f64`).
#[inline]
fn feature_vector(counts: BucketCounts) -> [f64; N_BUCKETS] {
    [
        counts[0] as f64,
        counts[1] as f64,
        counts[2] as f64,
        counts[3] as f64,
    ]
}

/// Stateless estimate based solely on the priors.
pub fn estimate_tokens_from_priors(input: &str) -> usize {
    if input.is_empty() {
        return 0;
    }
    let counts = classify_token_buckets(input);
    let mut sum = 0.0;
    for (i, &prior) in TOKEN_BUCKET_PRIORS.iter().enumerate() {
        sum += prior * counts[i] as f64;
    }
    (sum.round() as usize).max(1)
}

// -----------------------------------------------------------------------------
// Linear system solver (Gauss‑Jordan, small dense systems)
// -----------------------------------------------------------------------------

/// Solve the 4×4 linear system `A·x = b` using Gauss‑Jordan elimination
/// with partial pivoting. If a column is singular, the corresponding unknown
/// remains zero.
///
/// The matrix is expected to be symmetric positive‑definite in normal use,
/// but the solver handles ill‑conditioned cases gracefully.
pub fn solve_linear_system(a: Matrix4, b: Coefficients) -> Coefficients {
    let n = N_BUCKETS;
    // Build augmented matrix [A | b] as a mutable 4×5 array.
    let mut aug = [[0.0; N_BUCKETS + 1]; N_BUCKETS];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    for col in 0..n {
        // Find pivot
        let mut pivot = col;
        for r in (col + 1)..n {
            if aug[r][col].abs() > aug[pivot][col].abs() {
                pivot = r;
            }
        }
        if aug[pivot][col].abs() < 1e-12 {
            continue; // skip singular column
        }
        // Swap rows
        aug.swap(col, pivot);

        let diag = aug[col][col];
        // Eliminate all other rows
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[r][col] / diag;
            if factor.abs() < 1e-18 {
                continue;
            }
            for c in col..=n {
                aug[r][c] -= factor * aug[col][c];
            }
        }
    }

    let mut x = [0.0; N_BUCKETS];
    for i in 0..n {
        let diag = aug[i][i];
        x[i] = if diag.abs() < 1e-12 { 0.0 } else { aug[i][n] / diag };
    }
    x
}

// -----------------------------------------------------------------------------
// Snapshot and options
// -----------------------------------------------------------------------------

/// Serializable state of a calibrator.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TokenCalibratorSnapshot {
    pub a: Matrix4,
    pub g: Coefficients,
    pub strength: f64,
}

/// Check that a snapshot is structurally valid.
pub fn is_valid_snapshot(snap: &TokenCalibratorSnapshot) -> bool {
    if !snap.strength.is_finite() || snap.strength <= 0.0 {
        return false;
    }
    if snap.g.iter().any(|&v| !v.is_finite()) {
        return false;
    }
    for row in &snap.a {
        if row.iter().any(|&v| !v.is_finite()) {
            return false;
        }
    }
    true
}

/// Options for creating a calibrator.
#[derive(Debug, Clone)]
pub struct TokenCalibratorOptions {
    /// Prior strength (larger = stickier priors). Default `DEFAULT_PRIOR_STRENGTH`.
    pub prior_strength: Option<f64>,
    /// Exponential forgetting factor in `(0,1]`. `1.0` = no forgetting (default).
    pub forgetting: Option<f64>,
}

impl Default for TokenCalibratorOptions {
    fn default() -> Self {
        Self {
            prior_strength: None,
            forgetting: None,
        }
    }
}

// -----------------------------------------------------------------------------
// Main calibrator
// -----------------------------------------------------------------------------

/// Online ridge‑regression token estimator.
pub struct TokenCalibrator {
    a: Matrix4,           // Gram matrix (accumulated)
    g: Coefficients,      // right‑hand side (accumulated)
    gamma: f64,           // forgetting factor
    strength: f64,        // ridge penalty strength
    theta: Option<Coefficients>, // cached solution
}

impl TokenCalibrator {
    /// Create a new calibrator.
    ///
    /// If a valid `snapshot` is provided, it restores the state from it.
    /// Otherwise it starts with the priors seeded.
    pub fn new(opts: TokenCalibratorOptions, snapshot: Option<&TokenCalibratorSnapshot>) -> Self {
        let gamma = opts
            .forgetting
            .filter(|&f| f > 0.0 && f <= 1.0)
            .unwrap_or(1.0);

        // If we have a valid snapshot, use it directly.
        if let Some(snap) = snapshot {
            if is_valid_snapshot(snap) {
                return Self {
                    a: snap.a,
                    g: snap.g,
                    gamma,
                    strength: snap.strength,
                    theta: None,
                };
            }
        }

        // Otherwise start fresh with priors.
        let strength = opts
            .prior_strength
            .filter(|&s| s > 0.0)
            .unwrap_or(DEFAULT_PRIOR_STRENGTH);
        let mut calibrator = Self {
            a: [[0.0; N_BUCKETS]; N_BUCKETS],
            g: [0.0; N_BUCKETS],
            gamma,
            strength,
            theta: None,
        };
        calibrator.seed_priors();
        calibrator
    }

    /// Add the ridge prior (λ·I on the diagonal, λ·prior on the RHS).
    fn seed_priors(&mut self) {
        for j in 0..N_BUCKETS {
            self.a[j][j] += self.strength;
            self.g[j] += self.strength * TOKEN_BUCKET_PRIORS[j];
        }
    }

    /// Remove the ridge prior from the accumulated statistics.
    fn strip_priors(&mut self) {
        for j in 0..N_BUCKETS {
            self.a[j][j] -= self.strength;
            self.g[j] -= self.strength * TOKEN_BUCKET_PRIORS[j];
        }
    }

    /// Compute the coefficients (solves the normal equations).
    fn solve(&mut self) -> Coefficients {
        if let Some(theta) = self.theta {
            return theta;
        }
        let theta = solve_linear_system(self.a, self.g);
        self.theta = Some(theta);
        theta
    }

    /// Estimate the token count for the given input using the current model.
    pub fn estimate(&mut self, input: &str) -> usize {
        if input.is_empty() {
            return 0;
        }
        let counts = classify_token_buckets(input);
        let x = feature_vector(counts);
        let theta = self.solve();
        let mut sum = 0.0;
        // Floor coefficients at zero to avoid pathological negative rates.
        for j in 0..N_BUCKETS {
            sum += theta[j].max(0.0) * x[j];
        }
        (sum.round() as usize).max(1)
    }

    /// Incorporate one observed round: `input` whose real token count is `real_tokens`.
    pub fn observe(&mut self, input: &str, real_tokens: usize) {
        if real_tokens == 0 {
            return;
        }
        let counts = classify_token_buckets(input);
        let x = feature_vector(counts);
        // If all features are zero, this observation carries no information.
        if x.iter().all(|&v| v == 0.0) {
            return;
        }

        // Peel off the prior so that forgetting only affects the actual data.
        self.strip_priors();

        // Apply exponential forgetting if gamma < 1.0.
        if self.gamma < 1.0 {
            for i in 0..N_BUCKETS {
                self.g[i] *= self.gamma;
                for j in 0..N_BUCKETS {
                    self.a[i][j] *= self.gamma;
                }
            }
        }

        // Add the new observation: g += x * real_tokens, A += x * x^T
        let rt = real_tokens as f64;
        for i in 0..N_BUCKETS {
            self.g[i] += x[i] * rt;
            for j in 0..N_BUCKETS {
                self.a[i][j] += x[i] * x[j];
            }
        }

        // Re‑inject the prior.
        self.seed_priors();
        // Invalidate cached solution.
        self.theta = None;
    }

    /// Get the current learned rates per bucket.
    pub fn coefficients(&mut self) -> Coefficients {
        self.solve()
    }

    /// Capture the current state for persistence.
    pub fn snapshot(&self) -> TokenCalibratorSnapshot {
        TokenCalibratorSnapshot {
            a: self.a,
            g: self.g,
            strength: self.strength,
        }
    }

    // -------------------------------------------------------------------------
    // Model file loading (requires feature "serde")
    // -------------------------------------------------------------------------

    /// Parse a `models/models.json`-format string and return a calibrator
    /// seeded with the snapshot for `model_name`.
    ///
    /// Returns `None` if the model name is not found or the JSON is malformed.
    ///
    /// ```ignore
    /// let cal = TokenCalibrator::from_model("gpt-4o", JSON_STR).unwrap();
    /// ```
    #[cfg(feature = "serde")]
    pub fn from_model(name: &str, json_data: &str) -> Option<Self> {
        #[derive(serde::Deserialize)]
        struct ModelEntry {
            a: Matrix4,
            g: Coefficients,
            strength: f64,
        }
        #[derive(serde::Deserialize)]
        struct ModelsFile {
            models: std::collections::HashMap<String, ModelEntry>,
        }

        let file: ModelsFile = serde_json::from_str(json_data).ok()?;
        let entry = file.models.get(name)?;
        let snap = TokenCalibratorSnapshot {
            a: entry.a,
            g: entry.g,
            strength: entry.strength,
        };
        if !is_valid_snapshot(&snap) {
            return None;
        }
        Some(Self::new(TokenCalibratorOptions::default(), Some(&snap)))
    }

    /// Load a calibrator from the bundled `models/models.json` by model name.
    ///
    /// ```ignore
    /// let cal = TokenCalibrator::from_bundled_model("gpt-4o").unwrap();
    /// ```
    #[cfg(feature = "serde")]
    pub fn from_bundled_model(name: &str) -> Option<Self> {
        let json = include_str!("../../models/models.json");
        Self::from_model(name, json)
    }
}

// -----------------------------------------------------------------------------
// Tests (optional, just to ensure basic functionality)
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification() {
        let counts = classify_token_buckets("Hello 世界 123 😊");
        // "Hello" → 5 Latin, space → Latin (since 0x20 is included)
        // "世界" → 2 Han
        // " 123" → space + 3 digits
        // " 😊" → space + one emoji (other, 4 bytes)
        assert_eq!(counts[TokenBucket::Han as usize], 2);
        assert_eq!(counts[TokenBucket::Latin as usize], 8); // H,e,l,l,o + 3 spaces
        assert_eq!(counts[TokenBucket::Digit as usize], 3);
        // The emoji "😊" is 4 bytes, plus the preceding space is Latin, so other = 4
        assert_eq!(counts[TokenBucket::Other as usize], 4);
    }

    #[test]
    fn test_estimate_from_priors() {
        let n = estimate_tokens_from_priors("Hello world");
        // 11 chars (including space) * 0.25 ≈ 2.75 → round → 3, but max(1) -> 3
        assert_eq!(n, 3);
    }

    #[test]
    fn test_calibrator_basic() {
        // Use a small prior so observations dominate quickly
        let opts = TokenCalibratorOptions {
            prior_strength: Some(1.0),
            forgetting: None,
        };
        let mut cal = TokenCalibrator::new(opts, None);
        let est = cal.estimate("Hello world");
        assert_eq!(est, 3); // from prior

        // Observe a real token count (say 4 tokens for "Hello world")
        // Repeat so the estimate converges toward the observed value
        for _ in 0..20 {
            cal.observe("Hello world", 4);
        }
        let est2 = cal.estimate("Hello world");
        assert_eq!(est2, 4);
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let mut cal = TokenCalibrator::new(Default::default(), None);
        cal.observe("test", 2);
        let snap = cal.snapshot();
        let mut restored = TokenCalibrator::new(Default::default(), Some(&snap));
        assert_eq!(restored.a, cal.a);
        assert_eq!(restored.g, cal.g);
        assert_eq!(restored.strength, cal.strength);
        assert_eq!(restored.estimate("test"), cal.estimate("test"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_from_model() {
        let json = r#"{
            "models": {
                "test-model": {
                    "a": [[1000,0,0,0],[0,1000,0,0],[0,0,1000,0],[0,0,0,1000]],
                    "g": [1000,250,400,600],
                    "strength": 1000
                }
            }
        }"#;
        let mut cal = TokenCalibrator::from_model("test-model", json).unwrap();
        assert_eq!(cal.estimate("Hello world"), 3);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_from_model_missing() {
        let json = r#"{"models": {}}"#;
        assert!(TokenCalibrator::from_model("nonexistent", json).is_none());
    }
}