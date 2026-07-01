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
import { TokenCalibrator } from "@zlogic/token-calibrator";

const cal = new TokenCalibrator();

const estimate = cal.estimate(prompt);

// Feed the actual token count returned by your LLM provider.
cal.observe(prompt, actualTokens);

// Future estimates become more accurate.
```

### Python

```bash
pip install token-calibrator
```

```python
from token_calibrator import TokenCalibrator

cal = TokenCalibrator()
estimate = cal.estimate(prompt)

cal.observe(prompt, actual_tokens)
```

### Rust

```bash
cargo add token-calibrator
```

```rust
use token_calibrator::TokenCalibrator;

let mut cal = TokenCalibrator::new(Default::default(), None);

let estimate = cal.estimate(prompt);

cal.observe(prompt, actual_tokens);
```

### Go

```bash
go get github.com/zlogic/token-calibrator/go
```

```go
import "github.com/zlogic/token-calibrator/go/calibrator"

cal := calibrator.NewTokenCalibrator(calibrator.TokenCalibratorOptions{}, nil)
estimate := cal.Estimate(prompt)
cal.Observe(prompt, actualTokens)
```

---

## Examples

Run a complete walkthrough in your language of choice.

| Language       | Command                    |
| -------------- | -------------------------- |
| **Rust**       | `cargo run --example demo` |
| **TypeScript** | `npx tsx examples/demo.ts` |
| **Python**     | `python examples/demo.py`  |
| **Go**         | `go run ./cmd/demo`        |

Each demo:

1. Loads a model snapshot from `models.json` and estimates English, Chinese, and mixed-language text
2. Feeds real token counts back into the calibrator
3. Shows estimates improving over time
4. Prints the learned coefficients
5. Saves and restores a snapshot

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

Instead of starting from generic priors, you can initialize a calibrator with a
community-contributed snapshot trained for a specific model.

```python
import json

with open("models/models.json") as f:
    cal = TokenCalibrator.from_model("gpt-4o", f.read())
```

Snapshots provide a better starting point while still allowing the estimator to
continue learning from future observations.

Contributions of additional trained snapshots are welcome. See
[`CONTRIBUTING.md`](./CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](./LICENSE).