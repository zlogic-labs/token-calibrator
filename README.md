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
| Large vocabulary tables        | Four learned coefficients |
| Doesn't support unknown models | Works with future models  |

Instead of reproducing a tokenizer, token-calibrator learns the relationship
between text and token counts directly from your provider's responses.

---

## Features

* 🚀 Tokenizer-free
* 🤖 Works with any LLM
* 📈 Learns continuously from real API responses
* 🧠 Online ridge regression with configurable forgetting
* 💾 Tiny memory footprint (only four learned coefficients)
* 📦 Serializable model snapshots
* ⚡ Streaming-friendly
* 🌍 Available for TypeScript, Rust, Python, and Go

---

## How it works

```text
             Input text
                  │
                  ▼
      Character classification
 Han / Latin / Digit / Other
                  │
                  ▼
      Current coefficients
                  │
                  ▼
      Estimated token count
                  │
                  ▼
     observe(actual_tokens)
                  │
                  ▼
      Online ridge regression
```

Each observation adjusts the coefficients slightly, making future estimates
better match the tokenizer actually used by your model.

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
import { TokenTrainer, TokenEstimator, DEFAULT_MODELS_JSON } from "@zlogic/token-calibrator";

// ── Train from real usage ──
const trainer = new TokenTrainer();
trainer.observe(prompt, actualTokens);

// Export as complete models.json (merged with bundled defaults)
const json = trainer.toJsonMerged("my-model", DEFAULT_MODELS_JSON);

// ── Estimate tokens by model name ──
// Uses bundled default models (includes gpt-4o, claude-3.5-sonnet, ...)
const estimator = new TokenEstimator();
const estimate = estimator.estimate("gpt-4o", prompt);

// Or load your own:
const estimator2 = new TokenEstimator(jsonString);
```

### Python

```bash
pip install token-calibrator
```

```python
from token_calibrator import TokenTrainer, TokenEstimator, DEFAULT_MODELS_JSON

# Train
trainer = TokenTrainer()
trainer.observe(prompt, actual_tokens)

# Export as complete models.json (merged with bundled defaults)
json_str = trainer.to_json_merged("my-model", DEFAULT_MODELS_JSON)

# Estimate by model name — uses bundled default models
estimator = TokenEstimator()
estimate = estimator.estimate("gpt-4o", prompt)

# Or load custom JSON:
estimator2 = TokenEstimator(json_data)
```

### Rust

```bash
cargo add token-calibrator
```

```rust
use token_calibrator::{TokenTrainer, TokenEstimator, DEFAULT_MODELS_JSON};

// Train
let mut trainer = TokenTrainer::new(Default::default(), None);
trainer.observe(prompt, actual_tokens);

// Export as complete models.json (merged with bundled defaults)
let json = trainer.to_json_merged("my-model", DEFAULT_MODELS_JSON);

// Estimate by model name — uses bundled default models
let estimator = TokenEstimator::new(DEFAULT_MODELS_JSON).unwrap();
let estimate = estimator.estimate("gpt-4o", prompt);
```

### Go

```bash
go get github.com/zlogic/token-calibrator/go
```

```go
import "github.com/zlogic/token-calibrator/go/calibrator"

// Train
trainer := calibrator.NewTokenTrainer(calibrator.TokenTrainerOptions{}, nil)
trainer.Observe(prompt, actualTokens)

// Export as complete models.json (merged with bundled defaults)
jsonStr := trainer.ToJSONMerged("my-model", calibrator.DefaultModelsJSON)

// Estimate by model name — uses bundled default models
estimator := calibrator.NewTokenEstimatorFromJSON([]byte(calibrator.DefaultModelsJSON))
estimate, ok := estimator.Estimate("gpt-4o", prompt)
```

---

## Examples

Run a complete walkthrough in your language of choice.

| Language       | Command (train)                     | Command (estimate)                      |
| -------------- | ----------------------------------- | --------------------------------------- |
| **Rust**       | `cargo run --example train`         | `cargo run --example estimate [file]`   |
| **TypeScript** | `npx tsx examples/train.ts`         | `npx tsx examples/estimate.ts [file]`   |
| **Python**     | `python examples/train.py`          | `python examples/estimate.py [file]`    |
| **Go**         | `go run ./cmd/train`                | `go run ./cmd/estimate [file]`          |

The **train** example:
1. Creates a `TokenTrainer` and feeds hardcoded sample observations
2. Exports the learned coefficients as a JSON snapshot file (`trained-snapshot.json`)

The **estimate** example:
1. If a file argument is given, loads that JSON snapshot; otherwise uses the built-in default
2. Estimates token counts for English, Chinese, and mixed-language text

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

Characters are grouped into four buckets:

* **Han** (CJK ideographs)
* **Latin** (ASCII + Latin-1 Supplement)
* **Digit**
* **Other** (measured in UTF-8 bytes)

The coefficients are learned using online ridge regression by solving:

```text
(XᵀX + λI)α = Xᵀy
```

Because there are only four coefficients, the system solves a fixed 4×4 linear
system using Gauss–Jordan elimination. Exponential forgetting (`γ`) is
supported for streaming environments where tokenizer behavior may drift over
time.

---

## Pre-trained model snapshots

The library bundles community-contributed snapshots for various models
(`gpt-4o`, `claude-3.5-sonnet`, `gemini-2.0-flash`, etc.) via the
`DEFAULT_MODELS_JSON` constant. Just create an `TokenEstimator` with no arguments:

```python
from token_calibrator import TokenEstimator

# Bundled default models — no extra download needed
estimator = TokenEstimator()
estimator.estimate("gpt-4o", "Hello, world!")   # → ~3 tokens
estimator.estimate("claude-3.5-sonnet", text)    # → different model's estimate
```

The file `models/models.json` in the repo root is the source of truth.
Contributors update this file and the hardcoded `DEFAULT_MODELS_JSON` constants
in each language's code are updated accordingly.

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for how to contribute a snapshot.

---

## License

MIT — see [LICENSE](./LICENSE).