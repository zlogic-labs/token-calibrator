# token-calibrator 部署验收 & 基准测试报告

## 1. 多语言校准测试（基于 facebook/flores 200 数据集）

### 数据集
- **来源**: facebook/flores-200 dev split
- **语言**: 11 种（en/zh/ja/ko/ru/de/fr/es/hi/th/vi）
- **样本量**: 每语言 200 条，共 2200 条
- **真实值**: gpt-4o (tiktoken o200k_base)
- **划分**: 2/3 训练, 1/3 测试（分层抽样）

### 整体结果

| 方法 | MRE% | 相对 Prior 改进 |
|------|------|----------------|
| **Prior (通用先验)** | 100.2% | — |
| **Global calibrator** | **16.5%** | **+83.6%** |
| Adaptive (γ=0.97) | 46.2% | +53.9% |
| Per-language | 19.8% | +80.2% |

### 各语言表现（Global calibrator MRE%）

| 语言 | Prior | 校准后 | 说明 |
|------|-------|--------|------|
| **German (de)** | 13.6% | **9.0%** | 拉丁字母，与 prior 接近 |
| **French (fr)** | 12.0% | **9.3%** | 同上 |
| **Spanish (es)** | 12.5% | **8.6%** | 同上 |
| **English (en)** | 21.6% | **12.5%** | 显著改进 |
| **Vietnamese (vi)** | 41.5% | **13.6%** | 大幅改进 |
| **Chinese (zh)** | 27.8% | **15.8%** | 显著改进 |
| **Japanese (ja)** | 92.2% | **21.7%** | 大幅改进 |
| **Russian (ru)** | 70.0% | **23.9%** | 大幅改进 |
| **Hindi (hi)** | 344.6% | **24.7%** | 巨大改进 |
| **Arabic (ar)** | 211.5% | **12.4%** | 巨大改进 |
| **Thai (th)** | 307.4% | **9.8%** | 巨大改进 |
| **Korean (ko)** | 48.3% | **36.2%** | 改进较少（韩文 token 密度变化大） |

### 关键结论

1. **校准后误差从 100% 降至 16.5%** — 只用每语言 ~130 条训练样本
2. 拉丁字母语言（de/fr/es/en）误差最低（~8-12%），先验已经很准
3. 非拉丁语言从 prior 的 50-300% 误差降至 **10-36%**，改进极为显著
4. **Adaptive (forgetting) 表现更差** — flores 数据分布一致，不需要遗忘
5. Per-language 与 Global 接近 — 说明 global rates 已经很好地泛化

## 2. HuggingFace 数据集获取方式

```bash
# 需要 huggingface-cli (hf) 和 duckdb
pip install duckdb tiktoken token-calibrator

# 下载 flores-200 数据到 datasets/
hf download facebook/flores --repo-type dataset --local-dir datasets

# 运行校准 pipeline
python benchmark/flores_calibration.py
```

目前 `datasets/` 目录已包含 flores dev split 的 parquet 文件。
