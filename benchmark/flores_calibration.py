#!/usr/bin/env python3
"""
token-calibrator: Multi-language Calibration Pipeline
Reads flores parquet files, trains TokenCalibrator, evaluates vs tiktoken.

Usage: python flores_calibration.py
"""
import json, os, sys, random, statistics, subprocess
from pathlib import Path

# Install duckdb if missing
try:
    import duckdb
except ImportError:
    print("Installing duckdb...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "duckdb",
        "--trusted-host", "pypi.org", "--trusted-host", "files.pythonhosted.org"])
    import duckdb

import tiktoken
from token_calibrator import TokenCalibrator, estimate_tokens, TOKEN_BUCKET_PRIORS

SAMPLES = 200
TRAIN_RATIO = 2/3
OUT = Path(__file__).parent / "calibration_results"
PARQUET_FILE = Path(__file__).parent.parent / "datasets" / "dev-00000-of-00001.parquet"

# flores-200 uses {lang}_{script} column naming
LANGUAGES = [
    ("sentence_eng_Latn",    "en",  "English"),
    ("sentence_zho_Hans",    "zh",  "Chinese"),
    ("sentence_jpn_Jpan",    "ja",  "Japanese"),
    ("sentence_kor_Hang",    "ko",  "Korean"),
    ("sentence_rus_Cyrl",    "ru",  "Russian"),
    ("sentence_deu_Latn",    "de",  "German"),
    ("sentence_fra_Latn",    "fr",  "French"),
    ("sentence_spa_Latn",    "es",  "Spanish"),
    ("sentence_arb_Arab",   "ar",  "Arabic"),
    ("sentence_hin_Deva",    "hi",  "Hindi"),
    ("sentence_tha_Thai",    "th",  "Thai"),
    ("sentence_vie_Latn",    "vi",  "Vietnamese"),
]

def read_column(file_path, column, max_rows):
    con = duckdb.connect()
    sql = f'SELECT "{column}" FROM read_parquet(\'{file_path}\') WHERE LENGTH("{column}") > 10 LIMIT {max_rows}'
    result = con.execute(sql).fetchall()
    con.close()
    return [row[0].strip()[:2000] for row in result if row[0]]

def main():
    random.seed(42)
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Flores Calibration Pipeline")
    print("=" * 64)
    print(f"\n  Parquet: {PARQUET_FILE}")
    if not PARQUET_FILE.exists():
        print(f"  ERROR: File not found!"); sys.exit(1)

    # Read
    print("\n[1/5] Reading parquet columns...")
    all_texts = []; seen = set()
    for col, lang, name in LANGUAGES:
        sys.stdout.write(f"  {lang:4} {name:12} ")
        try:
            texts = read_column(str(PARQUET_FILE), col, SAMPLES)
            n = 0
            for t in texts:
                key = t[:60]
                if key in seen: continue
                seen.add(key)
                all_texts.append({"text": t, "lang": lang, "source": "flores"})
                n += 1
            print(f"{n} texts")
        except Exception as e:
            print(f"ERROR: {e}")

    # Stats
    lang_counts = {}
    for t in all_texts: lang_counts[t["lang"]] = lang_counts.get(t["lang"], 0) + 1
    print(f"\n  Total: {len(all_texts)} texts")
    for l, n in sorted(lang_counts.items()): print(f"    {l}: {n}")
    if not all_texts: print("ERROR: No data"); sys.exit(1)

    # Tokenize
    print("\n[2/5] Tokenizing with tiktoken (gpt-4o)...")
    enc = tiktoken.encoding_for_model("gpt-4o")
    for t in all_texts: t["real"] = len(enc.encode(t["text"]))
    
    vals = [t["real"] for t in all_texts]
    print(f"  Range: {min(vals)}-{max(vals)}, Mean: {statistics.mean(vals):.0f}")

    # Split
    print("\n[3/5] Splitting (2/3 train, 1/3 test)...")
    groups = {}
    for t in all_texts: groups.setdefault(t["lang"], []).append(t)
    train, test = [], []
    for items in groups.values():
        random.shuffle(items)
        n = max(1, int(len(items) * TRAIN_RATIO))
        train.extend(items[:n]); test.extend(items[n:])
    random.shuffle(train); random.shuffle(test)
    print(f"  Train: {len(train)}, Test: {len(test)}")

    # Train
    print("\n[4/5] Training...")
    cG = TokenCalibrator({"priorStrength": 1_000_000})
    cA = TokenCalibrator({"priorStrength": 1_000_000, "forgetting": 0.97})
    cL = {l: TokenCalibrator({"priorStrength": 1_000_000}) for l in groups}

    for t in train:
        cG.observe(t["text"], t["real"])
        cA.observe(t["text"], t["real"])
        if t["lang"] in cL: cL[t["lang"]].observe(t["text"], t["real"])

    print("  Global:", json.dumps({k: round(v, 4) for k, v in sorted(cG.rates().items())}))
    for l in sorted(cL.keys()):
        print(f"  {l}:", json.dumps({k: round(v, 4) for k, v in sorted(cL[l].rates().items())}))

    # Evaluate
    print(f"\n[5/5] Evaluating {len(test)} samples...\n")

    def eval_model(name, fn):
        bl = {}; errs = []
        for t in test:
            e = abs(fn(t) - t["real"]) / max(t["real"], 1)
            errs.append(e); bl.setdefault(t["lang"], []).append(e)
        return {"name": name, "mre": statistics.mean(errs) * 100,
                "by_lang": {k: statistics.mean(v) * 100 for k, v in sorted(bl.items())}}

    results = [
        eval_model("Prior", lambda t: estimate_tokens(t["text"], TOKEN_BUCKET_PRIORS)),
        eval_model("Global calibrator", lambda t: cG.estimate(t["text"])),
        eval_model("Adaptive (γ=0.97)", lambda t: cA.estimate(t["text"])),
        eval_model("Per-language", lambda t: cL.get(t["lang"], cG).estimate(t["text"])),
    ]

    print(f"{'Method':<25} {'MRE%':>8}"); print("-" * 35)
    for r in results: print(f"{r['name']:<25} {r['mre']:>7.1f}%")

    print(f"\n{'Per-language MRE%':-^70}")
    langs = sorted(groups.keys())
    header = "Lang".rjust(6) + "".join(r['name'][:14].rjust(16) for r in results)
    print(header); print("-" * len(header))
    for l in langs:
        print(l.rjust(6) + "".join(f"{r['by_lang'].get(l, 0):>15.1f}%" for r in results))

    print(f"\n{'Improvement vs Prior':-^40}")
    base = results[0]["mre"]
    for r in results[1:]:
        impr = (base - r["mre"]) / base * 100 if base > 0 else 0
        print(f"{r['name']:<25} {impr:>+7.1f}%")

    # Save
    report = {
        "config": {"samples": SAMPLES, "source": "flores dev split"},
        "dataset": {"total": len(all_texts), "train": len(train), "test": len(test), "by_lang": lang_counts},
        "trained_rates": {k: round(v, 4) for k, v in sorted(cG.rates().items())},
        "results": [{"label": r["name"], "mre": f"{r['mre']:.1f}%",
                      "by_lang": {k: f"{v:.1f}%" for k, v in r["by_lang"].items()}} for r in results],
    }
    p = OUT / "flores_calibration_report.json"
    with open(p, "w", encoding="utf-8") as f: json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport: {p}\nDone!")

if __name__ == "__main__":
    main()
