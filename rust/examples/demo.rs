//! Demo: token-calibrator in action — loading from bundled model snapshots.
//!
//! Shows how to initialise from a model in models.json, then learn and
//! snapshot.
//!
//! Run: `cargo run --example demo`

use token_calibrator::{TokenCalibrator, estimate_tokens_from_priors};

fn main() {
    println!("=== token-calibrator demo (Rust) ===\n");

    let english = "Hello, world! This is a test of the token calibrator.";
    let chinese = "你好世界，这是一个测试。";
    let mixed   = "Hello 你好 123 🎉";

    // 1. Load from bundled model file
    println!("── Load from bundled models.json ──");
    let mut cal = TokenCalibrator::from_bundled_model("gpt-4o")
        .unwrap_or_else(|| TokenCalibrator::from_bundled_model("default").unwrap());
    println!("Loaded model: gpt-4o");
    println!("English estimate: {} tokens", cal.estimate(english));
    println!("Coefficients    : {:.4?}", cal.coefficients());

    // 2. Compare with stateless priors
    println!("\n── Stateless priors (no calibration) ──");
    println!("English  : {:>4} chars → ~{:>3} tokens",
        english.len(), estimate_tokens_from_priors(english));
    println!("Chinese  : {:>4} chars → ~{:>3} tokens",
        chinese.chars().count(), estimate_tokens_from_priors(chinese));
    println!("Mixed    : {:>4} chars → ~{:>3} tokens",
        mixed.chars().count(), estimate_tokens_from_priors(mixed));

    // 3. Feed observations to adapt the model
    println!("\n── Training with real observations ──");
    cal.observe("Hello world", 3);
    cal.observe("你好世界", 6);

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
    println!("After {} observations:", 2 + more_data.len());
    println!("English estimate: {} tokens", cal.estimate(english));
    println!("Chinese estimate: {} tokens", cal.estimate(chinese));
    println!("Coefficients    : {:.4?}", cal.coefficients());

    // 4. Snapshot round-trip — load fresh from model and replay training
    let mut restored = TokenCalibrator::from_bundled_model("gpt-4o")
        .unwrap_or_else(|| TokenCalibrator::from_bundled_model("default").unwrap());
    // Feed the same data to the restored calibrator
    restored.observe("Hello world", 3);
    restored.observe("你好世界", 6);
    for (text, tokens) in &more_data {
        restored.observe(text, *tokens);
    }
    println!("\n── Restored from fresh model + same training ──");
    println!("English estimate: {} (match = {})",
        restored.estimate(english),
        restored.estimate(english) == cal.estimate(english));

    // 5. Save your own snapshot (to contribute back!)
    println!("\n── Your trained snapshot (ready to contribute!) ──");
    let trained = cal.snapshot();
    println!("a: {:?}", trained.a);
    println!("g: {:?}", trained.g);
    println!("strength: {}", trained.strength);

    println!("\n=== Demo complete ===");
}
