//! Demo: token-calibrator in action.
//!
//! Showcases prior estimates, learning from observations, and how the model
//! adapts its coefficients.
//!
//! Run: `cargo run --example demo`

use token_calibrator::{TokenCalibrator, TokenCalibratorOptions, estimate_tokens_from_priors};

fn main() {
    println!("=== token-calibrator demo (Rust) ===\n");

    // 1. Stateless prior estimate (no calibration required)
    let english = "Hello, world! This is a test of the token calibrator.";
    let chinese = "你好世界，这是一个测试。";
    let mixed   = "Hello 你好 123 🎉";

    println!("── Priors (no learning yet) ──");
    println!("English  : {:>4} chars → ~{:>3} tokens",
        english.len(), estimate_tokens_from_priors(english));
    println!("Chinese  : {:>4} chars → ~{:>3} tokens",
        chinese.chars().count(), estimate_tokens_from_priors(chinese));
    println!("Mixed    : {:>4} chars → ~{:>3} tokens",
        mixed.chars().count(), estimate_tokens_from_priors(mixed));

    // 2. Create a calibrator and learn
    let mut cal = TokenCalibrator::new(TokenCalibratorOptions::default(), None);
    println!("\n── Before observing ──");
    println!("English estimate: {} tokens", cal.estimate(english));
    println!("Coefficients    : {:?}", cal.coefficients());

    // 3. Feed two observations
    // Suppose "Hello world" tokenized as 3 tokens in a real model
    cal.observe("Hello world", 3);
    // A CJK-heavy prompt tokenized as 6 tokens
    cal.observe("你好世界", 6);

    println!("\n── After observing 2 rounds ──");
    println!("English estimate: {} tokens (was {})",
        cal.estimate(english), estimate_tokens_from_priors(english));
    println!("Coefficients    : {:.4?}", cal.coefficients());

    // 4. Feed more data → coefficients drift further from priors
    let more_data = [
        ("short", 2),
        ("a bit longer english text here", 8),
        ("more english words for the model to learn from", 12),
        ("中文中文中文中文", 12),
        ("1234567890", 4),
    ];
    for (text, tokens) in &more_data {
        cal.observe(text, *tokens);
    }
    println!("\n── After {} more observations ──", more_data.len());
    println!("English estimate: {} tokens", cal.estimate(english));
    println!("Chinese estimate: {} tokens", cal.estimate(chinese));
    println!("Coefficients    : {:.4?}", cal.coefficients());

    // 5. Snapshot / restore round-trip
    let snap = cal.snapshot();
    let mut restored = TokenCalibrator::new(TokenCalibratorOptions::default(), Some(&snap));
    println!("\n── Restored from snapshot ──");
    println!("English estimate: {} (match = {})",
        restored.estimate(english),
        restored.estimate(english) == cal.estimate(english));

    println!("\n=== Demo complete ===");
}
