# token-calibrator

**Calibrated, self-learning token estimator for live context-size display.**

No bundled tokenizer. Works with any LLM by learning per-character token rates
from real usage observations via online ridge regression.

---

## Core idea

Most context-size estimators hard-code a single token/char ratio (e.g., 0.25×
the character count). This is fragile — different models tokenize very
differently. The token-calibrator takes a different approach:

1. **Classify** each character into one of four buckets — **Han** (CJK
   ideographs), **Latin** (ASCII + Latin‑1 supplement), **Digit**, and
   **Other** (counted in UTF‑8 bytes).
2. **Learn** per-bucket token rates via **online ridge regression**, seeded
   with reasonable priors.
3. **Estimate** token count = Σ rate_bucket × count_bucket.
4. **Adapt** as you feed real token counts back via `observe()`.

The result: estimates converge toward whatever tokenizer the active model
actually uses, with no vendor-specific config and no bundled tokenizer.

## Languages

| Language | Package | Location |
|----------|---------|----------|
| **TypeScript** | npm (`token-calibrator`) | [`ts/`](./ts/) |
| **Python** | PyPI (`token-calibrator`) | [`python/`](./python/) |
| **Rust** | crates.io (`token-calibrator`) | [`rust/`](./rust/) |
| **Go** | `go get github.com/zlogic/token-calibrator/go` | [`go/`](./go/) |

## Quick start

### TypeScript
```ts
import { TokenCalibrator } from 'token-calibrator';

const cal = new TokenCalibrator();
console.log(cal.estimate("Hello 世界")); // 7 (from priors)

// Feed real token count from the provider
cal.observe("Hello 世界", 8);
console.log(cal.estimate("Hello 世界")); // 8 (adjusted)
```

### Rust
```rust
use token_calibrator::TokenCalibrator;

let mut cal = TokenCalibrator::new(Default::default(), None);
println!("{}", cal.estimate("Hello 世界")); // 7

cal.observe("Hello 世界", 8);
println!("{}", cal.estimate("Hello 世界")); // 8
```

### Go
```go
import "github.com/zlogic/token-calibrator/go/calibrator"

cal := calibrator.NewTokenCalibrator(...)
```

## Examples

Run a full walkthrough in your language of choice:

| Language | Command |
|----------|---------|
| **Rust** | `cargo run --example demo` (from `rust/`) |
| **TypeScript** | `npx tsx examples/demo.ts` (from `ts/`) |
| **Python** | `python examples/demo.py` (from `python/`) |
| **Go** | `go run ./cmd/demo` (from `go/`) |

Each demo:
1. Shows prior-based estimates for English, Chinese, and mixed text
2. Creates a `TokenCalibrator` and observes a few real token counts
3. Re-estimates the same strings to show the model adapting
4. Prints the learned per-bucket coefficients
5. Round-trips a snapshot to prove serialisation works

## Algorithm

The model is `tokens ≈ Σ_bucket α_bucket · count_bucket`, fit by solving the
normal equations `(X^T X + λI) α = X^T y`, where:

- `X` is the design matrix of bucket counts (one row per observed round),
- `y` is the vector of observed token counts,
- `λ` is the ridge penalty (prior strength).

Because the system is 4×4, the solve uses Gauss‑Jordan elimination directly
(inversion-free). Exponential forgetting (`γ`) is supported for streaming
environments where the model may drift.

## Pre-trained model snapshots

Instead of starting from generic priors, you can initialise a calibrator with
a **community-contributed snapshot** for your specific model. Snapshots are
stored in [`models/models.json`](./models/models.json).

```python
import json

with open("models/models.json") as f:
    cal = TokenCalibrator.from_model("gpt-4o", f.read())
```

> **Contribute your own trained snapshot!** See
> [`CONTRIBUTING.md`](./CONTRIBUTING.md) for instructions.

## License

MIT — see [LICENSE](./LICENSE).
