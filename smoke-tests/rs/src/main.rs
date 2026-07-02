/// Smoke test: token-calibrator (crates.io)
/// Run: cargo run
use std::collections::HashMap;
use token_calibrator::{
    TokenCalibrator, TokenCalibratorOptions, TokenEstimator, TokenEstimatorOptions,
    classify_token_buckets, estimate_tokens, token_bucket_priors, BUILTIN_TOKEN_RATES,
};

fn main() {
    // 1. Basic classify
    let c = classify_token_buckets("Hello 世界 123");
    println!("[RS] classifyTokenBuckets: {:?}", c);

    // 2. Prior estimate
    let priors = token_bucket_priors();
    let t1 = estimate_tokens("Hello world", &priors);
    println!("[RS] estimate (prior): {} (expected 3)", t1);

    // 3. Calibrate + estimate
    let mut cal = TokenCalibrator::new(TokenCalibratorOptions {
        prior_strength: Some(1_000.0),
        ..Default::default()
    });
    let observations: Vec<(&str, f64)> = vec![
        ("Hello world", 3.0),
        ("你好世界", 6.0),
        ("12345", 5.0),
    ];
    for (text, tokens) in &observations {
        cal.observe(text, *tokens);
    }
    let t2 = cal.estimate("Hello world");
    println!("[RS] calibrate + estimate: {} (expected ~3)", t2);

    // 4. Estimator with built-in models
    let est = TokenEstimator::new(None, TokenEstimatorOptions::default());
    for model in BUILTIN_TOKEN_RATES.keys().take(3) {
        let t = est.estimate(model, "Hello world");
        println!("[RS] {}: {} tokens", model, t);
    }

    // 5. Estimator with custom matrix
    let matrix = cal.to_matrix();
    let mut matrices = HashMap::new();
    matrices.insert("my-model".into(), matrix);
    let est2 = TokenEstimator::new(Some(matrices), TokenEstimatorOptions::default());
    println!("[RS] custom model estimate: {}", est2.estimate("my-model", "Hello world"));

    println!("[RS] All smoke tests passed!");
}
