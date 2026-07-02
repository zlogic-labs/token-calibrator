//! token-calibrator demo
//!
//! Demonstrates two usage modes:
//!   1. Train mode  — feed sample observations and export the accumulator
//!   2. Estimate mode — load trained data (or use built-in models) and estimate
//!
//! Usage:
//!   cargo run --example demo train              # train & export
//!   cargo run --example demo estimate           # estimate using built-in models
//!   cargo run --example demo estimate trained-snapshot.json  # use custom snapshot

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

use token_calibrator::{
    classify_token_buckets,
    TokenCalibrator, TokenCalibratorOptions, TokenEstimator, TokenEstimatorOptions,
    TokenAccumulator, BUILTIN_TOKEN_RATES,
};

// ───────────────────────── Train mode ─────────────────────────

fn cmd_train() {
    println!("=== Train mode ===\n");

    let mut cal = TokenCalibrator::new(TokenCalibratorOptions {
        prior_strength: Some(1_000.0),
        ..Default::default()
    });

    let observations: Vec<(&str, f64)> = vec![
        ("Hello world", 3.0),
        ("The quick brown fox jumps over the lazy dog", 10.0),
        ("你好，世界", 6.0),
        ("안녕하세요", 8.0),
        ("Привет мир", 5.0),
        ("123 456 7890", 6.0),
        ("🚀 Token estimation is amazing! 🎉", 12.0),
        ("Mixed 你好 Hello 123 😊", 9.0),
    ];

    for (text, tokens) in &observations {
        let counts = classify_token_buckets(text);
        println!("  observe: {:?}  →  {} tokens  ({:?})", counts, tokens, text);
        cal.observe(text, *tokens);
    }

    let rates = cal.rates();
    println!("\nLearned per-bucket rates:");
    let mut keys: Vec<&String> = rates.keys().collect();
    keys.sort();
    for k in keys {
        println!("  {:<10} {:.4}", k, rates[k]);
    }

    println!("\nEstimates after training:");
    let test_texts = ["Hello world", "你好", "Mixed 你好 123 😊"];
    for text in &test_texts {
        println!("  {:?}  →  {} tokens", text, cal.estimate(text));
    }

    let matrix = cal.to_matrix();
    let snapshot = serde_json::json!({
        "models": { "demo-model": matrix }
    });
    fs::write("trained-snapshot.json", serde_json::to_string_pretty(&snapshot).unwrap())
        .expect("failed to write snapshot");
    println!("\nExported accumulator to trained-snapshot.json\n");
}

// ───────────────────── Estimate mode ─────────────────────

fn cmd_estimate(snapshot_path: Option<&str>) {
    println!("=== Estimate mode ===\n");

    let matrices: HashMap<String, TokenAccumulator> = if let Some(path) = snapshot_path {
        if Path::new(path).exists() {
            let raw = fs::read_to_string(path).expect("failed to read snapshot");
            let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
            let models = parsed.get("models").unwrap_or(&parsed);
            let mut m = HashMap::new();
            if let Some(obj) = models.as_object() {
                for (name, val) in obj {
                    let acc: TokenAccumulator = serde_json::from_value(val.clone()).unwrap();
                    m.insert(name.clone(), acc);
                }
            }
            println!("Loaded {} model(s) from {}\n", m.len(), path);
            m
        } else {
            println!("File not found: {}, using built-in models\n", path);
            HashMap::new()
        }
    } else {
        println!("Using built-in default models (no snapshot file provided)\n");
        HashMap::new()
    };

    let est = TokenEstimator::new(Some(matrices), TokenEstimatorOptions::default());

    let test_texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "你好，世界",
        "안녕하세요",
        "Привет мир",
        "123 456 7890",
        "🚀 Token estimation is amazing! 🎉",
    ];

    let mut model_names: Vec<&String> = BUILTIN_TOKEN_RATES.keys().collect();
    model_names.truncate(5);
    let demo_model = "demo-model".to_string();
    model_names.push(&demo_model);

    for model in &model_names {
        println!("── {} ──", model);
        if est.has(model) {
            println!("  (user-calibrated data loaded)");
        }
        for text in &test_texts {
            let t = est.estimate(model, text);
            println!("  {:?}  →  {} tokens", text, t);
        }
        println!();
    }

    println!("── unknown-model (falls back to prior) ──");
    for text in &["Hello", "你好"] {
        let t = est.estimate("unknown-model", text);
        println!("  {:?}  →  {} tokens", text, t);
    }
}

// ─────────────────────── Main ───────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("estimate");
    let arg = args.get(2).map(|s| s.as_str());

    println!("token-calibrator demo (mode: {})\n", mode);

    match mode {
        "train" => cmd_train(),
        _ => cmd_estimate(arg),
    }
}
