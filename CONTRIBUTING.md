# Contributing to token-calibrator

## How to contribute a model snapshot

The calibrator learns per-bucket token rates from real usage. Once you've
trained a `TokenCalibrator` on enough real observations for a specific LLM,
you can **export your learned state** and share it with the community.

### 1. Train your calibrator

```python
from src.calibrator import TokenCalibrator

cal = TokenCalibrator(forgetting=0.98)  # gentle forgetting to adapt to drift

# Feed real (prompt, token_count) pairs from your LLM provider
for prompt, tokens in your_data:
    cal.observe(prompt, tokens)
```

> **Tip:** the more diverse your prompts (English, CJK, code, digits, emoji),
> the more accurate the snapshot will be across different content types.

### 2. Export the snapshot

```python
snap = cal.snapshot()
print(snap)
# {
#   'a': [[...], [...], [...], [...]],   # 4×4 Gram matrix
#   'g': [..., ..., ..., ...],           # 4-element RHS vector
#   'strength': 1000000.0,               # ridge strength
# }
```

### 3. Add it to `models/models.json`

Append a new entry under `"models"`:

```json
{
  "your-model-name": {
    "description": "Calibrated on ~1000 mixed prompts from …",
    "a": [[...], [...], [...], [...]],
    "g": [..., ..., ..., ...],
    "strength": 1000000.0
  }
}
```

Model name conventions:
- Use the **official model ID** as shown in the provider's API (e.g. `gpt-4o`,
  `claude-3.5-sonnet`, `gemini-2.0-flash`).
- Include a `description` with the model name, approximate number of
  observations, and the type of prompts used.

### 4. Open a Pull Request

- Make sure the JSON is valid (`python -m json.tool models/models.json`).
- Add a brief note about your training setup (how many rounds, what kinds of
  prompts, any special considerations).
- The CI will run the standard test suite and validate the JSON.

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

| Method / Function | Description |
|-------------------|-------------|
| `TokenCalibrator(opts, snapshot?)` | Constructor |
| `.estimate(input)` → `int` | Estimated token count |
| `.observe(input, real_tokens)` | Feed one observation |
| `.coefficients()` → `dict/array` | Learned per-bucket rates |
| `.snapshot()` → `Snapshot` | Serializable state |
| `.from_model(name, json_data)` → `Calibrator` | Load from models.json |
