# token-calibrator

**Tokenizer-free token estimation for any LLM.**

> Learn the tokenizer instead of shipping the tokenizer.

Instead of shipping a tokenizer, **token-calibrator** learns how your model
tokenizes text from real API usage. Estimates become more accurate over time,
without vendor-specific code or tokenizer updates.

Works with GPT, Claude, Gemini, DeepSeek, Qwen, Llama, Mistral, and future
models.

---

## Why?

Most LLM applications only need an approximate token count to:

* Display live context usage
* Estimate API cost
* Warn before exceeding context limits
* Budget prompts while editing

The traditional solution is to bundle a tokenizer for every supported model.
That works—but it comes with trade-offs.

| Traditional tokenizer          | token-calibrator          |
| ------------------------------ | ------------------------- |
| Bundled tokenizer              | No tokenizer              |
| Model-specific                 | Model-agnostic            |
| Requires tokenizer updates     | Learns automatically      |
| Large vocabulary tables        | Seven learned coefficients|
| Doesn't support unknown models | Works with future models  |

Instead of reproducing a tokenizer, token-calibrator learns the relationship
between text and token counts directly from your provider's responses.

---

## Features

* 🚀 Tokenizer-free
* 🤖 Works with any LLM
* 📈 Learns continuously from real API responses
* 🧠 Online ridge regression with configurable forgetting
* 💾 Tiny memory footprint (only seven learned coefficients)
* 📦 Serializable accumulator state
* ⚡ Streaming-friendly
* 🌍 Available for TypeScript, Rust, Python, and Go

---

## How it works

```text
             Input text
                  │
                  ▼
      Character classification
 Han / Latin / Digit / Hangul /
  Cyrillic / Emoji / Other
                  │
                  ▼
      Current per-bucket rates
                  │
                  ▼
      Estimated token count
                  │
                  ▼
     observe(real_tokens)
                  │
                  ▼
      Online ridge regression
```

Each observation adjusts the coefficients slightly, making future estimates
better match the tokenizer actually used by your model.

---

## Buckets

Characters are grouped into **seven** buckets to capture how different scripts
tokenize:

| Bucket    | Count unit    | Scripts / characters                                    | Typical tokens/unit |
|-----------|---------------|--------------------------------------------------------|---------------------|
| Han       | character     | CJK ideographs (Chinese, Japanese Kanji)               | ~0.9–1.3            |
| Latin     | character     | ASCII, Latin-1 Supplement, punctuation, spaces          | ~0.23–0.25          |
| Digit     | character     | 0–9                                                    | ~0.4–1.3            |
| Hangul    | character     | Korean Hangul syllables + Jamo                          | ~0.8–1.5            |
| Cyrillic  | character     | Cyrillic + Supplement                                   | ~0.3–0.5            |
| Emoji     | character     | Emoticons, pictographs, flags, dingbats                 | ~1.4–3.2            |
| Other     | UTF-8 byte    | Everything else (kana, Arabic, Thai, control chars...)  | ~0.16–0.40          |

Every bucket is counted in **characters** except `other`, which is counted in
**UTF-8 bytes** — scripts outside a tokenizer's merge vocabulary fall back to
byte-level BPE (≈ 1 token/byte).

---

## Languages

| Language       | Package                                        | Install                                | Location               |
| -------------- | ---------------------------------------------- | -------------------------------------- | ---------------------- |
| **TypeScript** | npm (`@zlogic/token-calibrator`)               | `npm install @zlogic/token-calibrator` | [`ts/`](./ts/)         |
| **Python**     | PyPI (`token-calibrator`)                      | `pip install token-calibrator`         | [`python/`](./python/) |
| **Rust**       | crates.io (`token-calibrator`)                 | `cargo add token-calibrator`           | [`rust/`](./rust/)     |
| **Go**         | `go get github.com/zlogic/token-calibrator/go` | `go get github.com/zlogic/...`         | [`go/`](./go/)         |

---

## Quick start

### TypeScript

```bash
npm install @zlogic/token-calibrator
```

```ts
import { TokenCalibrator, TokenEstimator } from "@zlogic/token-calibrator";

// ── Train from real usage ──
const cal = new TokenCalibrator();
cal.observe(prompt, actualTokens);
const matrix = cal.toMatrix(); // save per model

// ── Estimate tokens by model name ──
// Uses built-in baseline (includes gpt-4o, deepseek-chat, llama-3.1-70b, ...)
const estimator = new TokenEstimator({ "my-model": matrix });
const tokens = estimator.estimate("gpt-4o", prompt);
```

### Python

```bash
pip install token-calibrator
```

```python
from token_calibrator import TokenCalibrator, TokenEstimator

# Train
cal = TokenCalibrator()
cal.observe(prompt, actual_tokens)
matrix = cal.to_matrix()  # save per model

# Estimate by model name — uses built-in baseline models
estimator = TokenEstimator({"my-model": matrix})
tokens = estimator.estimate("gpt-4o", prompt)
```

### Rust

```bash
cargo add token-calibrator
```

```rust
use token_calibrator::{TokenCalibrator, TokenCalibratorOptions, TokenEstimator, TokenEstimatorOptions};
use std::collections::HashMap;

// Train
let mut cal = TokenCalibrator::new(TokenCalibratorOptions::default());
cal.observe(prompt, actual_tokens);
let matrix = cal.to_matrix();  // clone for persistence

// Estimate by model name — uses built-in baseline models
let mut matrices = HashMap::new();
matrices.insert("my-model".into(), matrix);
let estimator = TokenEstimator::new(Some(matrices), TokenEstimatorOptions::default());
let tokens = estimator.estimate("gpt-4o", prompt);
```

### Go

```bash
go get github.com/zlogic/token-calibrator/go
```

```go
import calibrator "github.com/zlogic/token-calibrator/go"

// Train
cal := calibrator.NewTokenCalibrator(calibrator.TokenCalibratorOptions{})
cal.Observe(prompt, actualTokens)
matrix := cal.ToMatrix()

// Estimate by model name — uses built-in baseline models
matrices := map[string]calibrator.TokenAccumulator{"my-model": matrix}
estimator := calibrator.NewTokenEstimator(matrices, calibrator.TokenEstimatorOptions{})
tokens := estimator.Estimate("gpt-4o", prompt)
```

---

## Examples

Run a complete walkthrough in your language of choice. The demo has two modes:

- **calibrate** — feeds sample observations, shows learned rates, exports accumulator to `calibrated-snapshot.json`
- **estimate** — loads calibrated data (or uses built-in models) and estimates token counts for several text types

| Language       | Command (calibrate)                          | Command (estimate)                                  |
| -------------- | -------------------------------------------- | --------------------------------------------------- |
| **TypeScript** | `npx tsx examples/demo.ts calibrate`         | `npx tsx examples/demo.ts estimate [file]`          |
| **Python**     | `python examples/demo.py calibrate`          | `python examples/demo.py estimate [file]`           |
| **Rust**       | `cargo run --example demo calibrate`         | `cargo run --example demo estimate [file]`          |
| **Go**         | `go run ./cmd/demo calibrate`                | `go run ./cmd/demo estimate [file]`                 |

The **calibrate** output:
1. Shows each observation with its per-bucket classification
2. Prints the learned per-bucket rates
3. Shows estimated tokens for a few test strings
4. Exports the accumulator as `trained-snapshot.json`

The **estimate** output:
1. If a file argument is given, loads that JSON snapshot; otherwise uses built-in models only
2. Shows estimated tokens for several text types across different models

---

## Use cases

Ideal for applications that need fast, lightweight token estimation:

* AI chat clients
* Agent frameworks
* IDE extensions
* Prompt editors
* Live context meters
* Cost estimation
* Token budgeting
* Streaming interfaces

---

## Algorithm

The estimator models:

```text
tokens ≈ Σ αᵢ × bucket_countᵢ
```

Characters are grouped into **seven** buckets:

* **Han** (CJK ideographs)
* **Latin** (ASCII + Latin-1 Supplement)
* **Digit**
* **Hangul**
* **Cyrillic**
* **Emoji**
* **Other** (measured in UTF-8 bytes)

Every bucket is a per-character rate, except `other` which is a per-byte rate.
The coefficients are learned using online ridge regression by solving:

```text
(XᵀX + λI)α = Xᵀy
```

Because there are only seven coefficients, the system solves a fixed 7×7 linear
system using Gauss–Jordan elimination. Exponential forgetting (`γ`) is
supported for streaming environments where tokenizer behavior may drift over
time.

---

## API reference

### TokenCalibrator (writer / trainer)

| Method                                          | Description                          |
| ----------------------------------------------- | ------------------------------------ |
| `TokenCalibrator(opts?)`                        | Create a calibrator                  |
| `.observe(input, real_tokens)`                  | Feed one observation                 |
| `.rates()` → `TokenRates`                       | Current learned per-bucket rates     |
| `.estimate(input)` → `number`                   | Estimate tokens using current rates  |
| `.to_matrix()` → `TokenAccumulator`             | Raw data-only accumulator (persist)  |

### TokenEstimator (reader)

| Method                                                   | Description                                |
| -------------------------------------------------------- | ------------------------------------------ |
| `TokenEstimator(matrices?, opts?)`                        | Create estimator from model matrices       |
| `.estimate(model_name, input)` → `number`                 | Estimate tokens by model name              |
| `.rates(model_name)` → `TokenRates`                       | Effective rates for a model                |
| `.has(model_name)` → `bool`                               | Whether model has user-calibrated data     |

### Pure functions

| Function                                                    | Description                            |
| ----------------------------------------------------------- | -------------------------------------- |
| `classify_token_buckets(input)` → `counts`                  | Classify text into per-bucket counts   |
| `feature_vector(counts)` → `[f64; 7]`                       | Ordered feature vector for regression  |
| `estimate_tokens(input, rates)` → `number`                  | Pure token estimate                    |
| `empty_accumulator()` → `TokenAccumulator`                  | Fresh zero-initialised accumulator     |
| `accumulate(acc, input, real_tokens, forgetting?)` → `acc`  | Fold one observation (pure)            |
| `rates_from_accumulator(acc, priorStrength?, prior?)` → `...` | Solve ridge regression                |
| `is_valid_accumulator(acc)` → `bool`                        | Structural validity check              |
| `solve_linear_system(a, b)` → `[f64; 7]`                    | Gauss–Jordan elimination               |
| `derive_rates(matrices, ...)` → `{model: rates}`            | Batch derive rates from multiple accs  |

### Constants

| Constant                         | Type                        | Description                          |
| -------------------------------- | --------------------------- | ------------------------------------ |
| `TOKEN_BUCKETS`                  | `string[]`                  | Ordered bucket names                 |
| `N_BUCKETS`                      | `number` / `usize` / `int`  | Number of buckets (7)                |
| `TOKEN_BUCKET_PRIORS`            | `TokenRates`                | Default generic priors               |
| `DEFAULT_PRIOR_STRENGTH`         | `1_000_000`                 | Default ridge strength λ             |
| `BUILTIN_TOKEN_RATES`            | `{model: TokenRates}`       | Shipped baseline for known models    |

---

## License

MIT — see [LICENSE](./LICENSE).
