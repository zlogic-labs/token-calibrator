# Contributing to token-calibrator

## How to contribute a model snapshot

The calibrator learns per-bucket token rates from real usage. Once you've
trained a `TokenCalibrator` on enough real observations for a specific LLM,
you can **export the accumulator** (`to_matrix()`) and share it with the
community.

### 1. Train your calibrator

```python
from token_calibrator import TokenCalibrator

cal = TokenCalibrator({"forgetting": 0.98})

# Feed real (prompt, token_count) pairs from your LLM provider
for prompt, tokens in your_data:
    cal.observe(prompt, tokens)
```

> **Tip:** the more diverse your prompts (English, CJK, code, digits, emoji),
> the more accurate the snapshot will be across different content types.

### 2. Export the accumulator

```python
matrix = cal.to_matrix()
print(matrix)
# {
#   'a': [[...], [...], ...],  # 7×7 Gram matrix
#   'g': [..., ..., ...],      # 7-element RHS vector
# }
```

The accumulator is a pure data sum — no prior baked in. It's compact and can
be stored as JSON.

### 3. Submit your snapshot

Open a pull request that adds your trained matrix to the `BUILTIN_TOKEN_RATES`
constant in each language's codebase:

| Language       | File                                           |
| -------------- | ---------------------------------------------- |
| **TypeScript** | `ts/src/builtin-rates.ts`                      |
| **Python**     | `python/src/token_calibrator/builtin_rates.py` |
| **Rust**       | `rust/src/builtin_rates.rs`                    |
| **Go**         | `go/builtin_rates.go`                          |

Model name conventions:
- Use the **official model ID** as shown in the provider's API (e.g. `gpt-4o`,
  `deepseek-chat`, `llama-3.1-70b`).
- Include a comment with the model name, approximate number of observations,
  and the type of prompts used.

### 4. Open a Pull Request

- Make sure the code compiles and all tests pass in each language.
- Add a brief note about your training setup (how many rounds, what kinds of
  prompts, any special considerations).
- The CI will run the standard test suite across all languages.

---

## Development setup

### TypeScript
```bash
cd ts
npm install
npm test
```

### Python
```bash
cd python
pip install -e ".[dev]"
pytest
```

### Rust
```bash
cd rust
cargo test
```

### Go
```bash
cd go
go test ./...
```

---

## Code style

- **Rust**: `cargo clippy` clean, standard `rustfmt`.
- **TypeScript**: Standard Prettier + ESLint config (if added).
- **Python**: Follow PEP 8, type annotations for all public functions.
- **Go**: `gofmt` + `go vet` clean.

## Cross-language consistency

All four implementations should expose the same public API:

| Method / Function                          | Description                              |
| ------------------------------------------ | ---------------------------------------- |
| `TokenCalibrator(opts?)`                   | Constructor — learns from observations   |
| `.observe(input, real_tokens)`             | Feed one observation                     |
| `.rates()` → `TokenRates`                  | Learned per-bucket rates                 |
| `.estimate(input)` → `number`              | Token estimate using current rates       |
| `.to_matrix()` → `TokenAccumulator`        | Serializable accumulator (for export)    |
| `TokenEstimator(matrices?, opts?)`         | Constructor — uses built-in baseline     |
| `.estimate(model_name, input)` → `number`  | Estimated token count by model name      |
| `.rates(model_name)` → `TokenRates`        | Effective rates for a model              |
| `.has(model_name)` → `bool`                | Whether model has user-calibrated data   |
| `estimate_tokens(input, rates)` → `number` | Stateless token estimate                 |
| `BUILTIN_TOKEN_RATES`                      | Shipped baseline for known models        |

## Running the demo

```bash
# TypeScript
npx tsx ts/examples/demo.ts train
npx tsx ts/examples/demo.ts estimate

# Python
python python/examples/demo.py train
python python/examples/demo.py estimate

# Rust
cargo run --manifest-path rust/Cargo.toml --example demo train
cargo run --manifest-path rust/Cargo.toml --example demo estimate

# Go
go run ./go/cmd/demo train
go run ./go/cmd/demo estimate
```
