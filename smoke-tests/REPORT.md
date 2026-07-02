# token-calibrator 部署验收 & 基准测试

## 1. Smoke Tests — 各语言部署验证

所有 4 种语言的包都已成功发布并可从官方仓库安装：

| 语言 | 包管理器 | 包名 | 版本 | 状态 |
|------|----------|------|------|------|
| **TypeScript** | npm | `@zlogic/token-calibrator` | 1.0.1 | ✅ |
| **Python** | PyPI | `token-calibrator` | 1.0.1 | ✅ |
| **Rust** | crates.io | `token-calibrator` | 1.0.1 | ✅ |
| **Go** | GitHub | `github.com/zlogic-libs/token-calibrator/go` | v1.0.0+ | ✅ |

运行 smoke test：

```bash
# TS
cd smoke-tests/ts && npm install && node index.js

# Python
cd smoke-tests/py && pip install token-calibrator && python main.py

# Rust
cd smoke-tests/rs && cargo run

# Go
cd smoke-tests/go && go run main.go
```

---

## 2. 基准测试报告

> 使用 `tiktoken` 获取 3 个模型（gpt-4o / gpt-3.5-turbo / gpt-4）对 19 个涵盖 7 个 bucket 的文本的真实 token 数，
> 与 `token-calibrator` 内置模型的估计值进行对比。

### 整体汇总

| 指标 | Calibrator | Prior-only | 改进 |
|------|-----------|-----------|------|
| **平均误差** | **24.9%** | 32.5% | **+23%** |
| **中位误差** | **20.0%** | 22.2% | **+10%** |
| **最大误差** | **100.0%** | 200.0% | **+50%** |
| 样本更优数 | **33/57** | — | — |

### 各模型表现

| 模型 | 编码器 | Calibrator 平均误差 | Prior 平均误差 |
|------|--------|-------------------|---------------|
| **gpt-4o** | o200k_base | **33.2%** | 56.6% |
| **gpt-3.5-turbo** | cl100k_base | **22.0%** | 24.7% |
| **gpt-4** | cl100k_base | **19.5%** | 22.2% |

### 观察结论

1. **内置模型优于通用先验**：对于所有测试模型，calibrator 的内置 rates 都比通用 `TOKEN_BUCKET_PRIORS` 更准确，平均误差降低 23%
2. **CJK 文本误差较大**：中文（han bucket）的 token 密度变化大，calibrator 使用固定 rate 难以精确匹配，但比 prior 已有显著改进
3. **Emoji 和 Hangul 独立 bucket 有效**：将 emoji/hangul/cyrillic 独立出来防止了跨脚本的相互污染
4. **校准后会更准**：用户通过 `observe()` 提供真实数据后，regression 会逐步收敛到实际 tokenizer 行为

### 运行基准测试

```bash
cd benchmark && npm install && node benchmark.js
```
