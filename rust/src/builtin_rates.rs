//! Built-in per-model token rates (shipped baseline).

use std::collections::HashMap;
use once_cell::sync::Lazy;

use crate::buckets::TokenRates;

fn make_rates(vals: &[(f64, f64, f64, f64, f64, f64, f64)]) -> Vec<TokenRates> {
    vals.iter()
        .map(|&(han, latin, digit, hangul, cyrillic, emoji, other)| {
            let mut m = HashMap::new();
            m.insert("han".into(), han);
            m.insert("latin".into(), latin);
            m.insert("digit".into(), digit);
            m.insert("hangul".into(), hangul);
            m.insert("cyrillic".into(), cyrillic);
            m.insert("emoji".into(), emoji);
            m.insert("other".into(), other);
            m
        })
        .collect()
}

pub static BUILTIN_TOKEN_RATES: Lazy<HashMap<String, TokenRates>> = Lazy::new(|| {
    let names = [
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5-mini",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "llama-3.1-70b",
        "llama-3.1-8b",
        "mistral-7b",
        "mistral-small",
        "o1",
        "o3",
        "qwen-max",
        "qwen-plus",
        "qwen2.5-72b",
        "qwen2.5-7b",
    ];
    let deepseek = (0.6896, 0.2521, 0.838, 0.8542, 0.2955, 2.0485, 0.1799);
    let gpt_3 = (1.2812, 0.2458, 0.7706, 1.3339, 0.5288, 2.9113, 0.3672);
    let gpt_4_1 = (0.9369, 0.2272, 0.8867, 0.7902, 0.2831, 2.2016, 0.1561);
    let llama = (0.8905, 0.2439, 0.7827, 0.7994, 0.3178, 2.9855, 0.1863);
    let mistral = (1.1036, 0.2798, 1.2948, 1.4589, 0.3861, 3.1633, 0.3988);
    let qwen = (0.6907, 0.2588, 1.177, 0.8754, 0.3571, 1.3806, 0.2598);

    let all_rates: Vec<TokenRates> = make_rates(&[
        deepseek, deepseek, deepseek, deepseek,
        gpt_3, gpt_3, gpt_3,
        gpt_4_1, gpt_4_1, gpt_4_1, gpt_4_1,
        gpt_4_1, gpt_4_1, gpt_4_1, gpt_4_1,
        llama, llama,
        mistral, mistral,
        gpt_4_1, gpt_4_1, // o1, o3
        qwen, qwen, qwen, qwen,
    ]);

    let mut map = HashMap::new();
    for (name, rates) in names.iter().zip(all_rates) {
        map.insert(name.to_string(), rates);
    }
    map
});
