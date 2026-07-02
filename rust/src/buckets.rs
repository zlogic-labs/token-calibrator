//! Bucketing + pure estimation — the model-blind feature layer.
//!
//! A text is classified into per-bucket counts (`classify_token_buckets`), then
//! turned into a token estimate given a set of per-bucket rates (`estimate_tokens`).
//! Both are pure and stateless; the fitting of rates lives in `calibration`,
//! the stateful wrappers in `calibrator` / `estimator`.
//!
//! Buckets: Han / Latin / Digit / Hangul / Cyrillic / Emoji / Other. Every
//! bucket is a CHARACTER count except `other`, which is a UTF-8 BYTE count —
//! scripts outside a tokenizer's merge vocab fall back to byte-level BPE
//! (≈ 1 token/byte), so byte count tracks their cost better than char count.

use std::collections::HashMap;

/// Ordered bucket names.
pub const TOKEN_BUCKETS: &[&str] = &[
    "han", "latin", "digit", "hangul", "cyrillic", "emoji", "other",
];

/// Number of buckets — the dimension of the accumulator / feature vector.
pub const N_BUCKETS: usize = 7;

/// Per-bucket token rates.
pub type TokenRates = HashMap<String, f64>;

/// Default per-bucket priors (the ridge target when a model has no / little data).
pub fn token_bucket_priors() -> TokenRates {
    let mut m = HashMap::new();
    m.insert("han".into(), 1.0);
    m.insert("latin".into(), 0.25);
    m.insert("digit".into(), 0.4);
    m.insert("hangul".into(), 1.2);
    m.insert("cyrillic".into(), 0.5);
    m.insert("emoji".into(), 1.5);
    m.insert("other".into(), 0.6);
    m
}

fn utf8_len(cp: u32) -> u32 {
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

/// Single-pass classification into the buckets. Every bucket is a CHARACTER
/// count except `other`, which is a UTF-8 BYTE count.
pub fn classify_token_buckets(input: &str) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = [
        ("han".into(), 0u32),
        ("latin".into(), 0),
        ("digit".into(), 0),
        ("hangul".into(), 0),
        ("cyrillic".into(), 0),
        ("emoji".into(), 0),
        ("other".into(), 0),
    ]
    .into_iter()
    .collect();

    if input.is_empty() {
        return counts;
    }

    for ch in input.chars() {
        let cp = ch as u32;

        if (0x30..=0x39).contains(&cp) {
            *counts.get_mut("digit").unwrap() += 1;
        } else if is_han(cp) {
            *counts.get_mut("han").unwrap() += 1;
        } else if (0x20..=0x024F).contains(&cp) {
            *counts.get_mut("latin").unwrap() += 1;
        } else if is_hangul(cp) {
            *counts.get_mut("hangul").unwrap() += 1;
        } else if is_cyrillic(cp) {
            *counts.get_mut("cyrillic").unwrap() += 1;
        } else if is_emoji(cp) {
            *counts.get_mut("emoji").unwrap() += 1;
        } else {
            *counts.get_mut("other").unwrap() += utf8_len(cp);
        }
    }

    counts
}

fn is_han(cp: u32) -> bool {
    (0x3400..=0x4DBF).contains(&cp)
        || (0x4E00..=0x9FFF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2EBEF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
        || (0x30000..=0x323AF).contains(&cp)
}

fn is_hangul(cp: u32) -> bool {
    (0xAC00..=0xD7A3).contains(&cp)
        || (0x1100..=0x11FF).contains(&cp)
        || (0x3130..=0x318F).contains(&cp)
}

fn is_cyrillic(cp: u32) -> bool {
    (0x0400..=0x04FF).contains(&cp) || (0x0500..=0x052F).contains(&cp)
}

fn is_emoji(cp: u32) -> bool {
    (0x1F000..=0x1FAFF).contains(&cp)
        || (0x2600..=0x26FF).contains(&cp)
        || (0x2700..=0x27BF).contains(&cp)
}

/// Ordered feature vector for the regression: bucket counts in `TOKEN_BUCKETS` order.
pub fn feature_vector(counts: &HashMap<String, u32>) -> [f64; N_BUCKETS] {
    let mut v = [0.0f64; N_BUCKETS];
    for (i, bucket) in TOKEN_BUCKETS.iter().enumerate() {
        v[i] = *counts.get(*bucket).unwrap_or(&0) as f64;
    }
    v
}

/// Pure estimate: input + per-bucket rates → token count. Model-blind.
pub fn estimate_tokens(input: &str, rates: &TokenRates) -> u32 {
    if input.is_empty() {
        return 0;
    }
    let counts = classify_token_buckets(input);
    let mut sum = 0.0f64;
    for bucket in TOKEN_BUCKETS {
        let rate = rates.get(*bucket).copied().unwrap_or(0.0).max(0.0);
        let cnt = *counts.get(*bucket).unwrap_or(&0) as f64;
        sum += rate * cnt;
    }
    (sum.round().max(1.0)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_latin() {
        let c = classify_token_buckets("Hello");
        assert_eq!(c["han"], 0);
        assert_eq!(c["latin"], 5);
        assert_eq!(c["digit"], 0);
    }

    #[test]
    fn test_classify_chinese() {
        let c = classify_token_buckets("你好");
        assert_eq!(c["han"], 2);
        assert_eq!(c["latin"], 0);
    }

    #[test]
    fn test_classify_empty() {
        let c = classify_token_buckets("");
        for b in TOKEN_BUCKETS {
            assert_eq!(c[*b], 0);
        }
    }

    #[test]
    fn test_classify_korean() {
        let c = classify_token_buckets("안녕하세요");
        assert_eq!(c["hangul"], 5);
    }

    #[test]
    fn test_classify_cyrillic() {
        let c = classify_token_buckets("Привет");
        assert_eq!(c["cyrillic"], 6);
    }

    #[test]
    fn test_classify_emoji() {
        let c = classify_token_buckets("🚀🎉");
        assert_eq!(c["emoji"], 2);
    }

    #[test]
    fn test_feature_vector_order() {
        let mut counts = HashMap::new();
        counts.insert("han".into(), 2);
        counts.insert("latin".into(), 5);
        counts.insert("digit".into(), 3);
        counts.insert("hangul".into(), 0);
        counts.insert("cyrillic".into(), 0);
        counts.insert("emoji".into(), 1);
        counts.insert("other".into(), 4);
        let fv = feature_vector(&counts);
        assert_eq!(fv, [2.0, 5.0, 3.0, 0.0, 0.0, 1.0, 4.0]);
    }

    #[test]
    fn test_estimate_prior_hello_world() {
        let priors = token_bucket_priors();
        assert_eq!(estimate_tokens("Hello world", &priors), 3);
    }

    #[test]
    fn test_estimate_empty() {
        let priors = token_bucket_priors();
        assert_eq!(estimate_tokens("", &priors), 0);
    }

    #[test]
    fn test_estimate_floor_at_1() {
        let priors = token_bucket_priors();
        assert_eq!(estimate_tokens("a", &priors), 1);
    }
}
