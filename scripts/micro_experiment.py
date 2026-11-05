"""
Phase 2 — what does the simplified micro feature set cost?

Runs the same split / CV / hold-out procedure as baseline.py on three feature sets:
    full           all 298 engineered features (baseline)
    micro          curated micro profile + bureau-score proxy (EXT_SOURCE_1/2/3)
    micro_no_ext   curated micro profile without the anonymised scores

Usage:
    python scripts/micro_experiment.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from baseline import print_summary, run_models  # noqa: E402
from micro_profile import build_micro_features  # noqa: E402


def main() -> None:
    t0 = time.time()
    df = pd.read_csv(ROOT / "data" / "processed" / "train_features.csv")
    y = df["TARGET"]

    variants = {
        "micro": build_micro_features(df, include_bureau_score=True),
        "micro_no_ext": build_micro_features(df, include_bureau_score=False),
    }
    results = {"full": json.loads((ROOT / "reports" / "baseline_metrics.json").read_text())}
    for name, X in variants.items():
        print(f"\n=== {name} ===", flush=True)
        results[name] = run_models(X, y, t0)
        print_summary(results[name])

    out = ROOT / "reports" / "micro_experiment.json"
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 64)
    print(f"{'Feature set':<14}{'#feat':>6}{'Scorecard AUC':>15}{'XGBoost AUC':>13}")
    for name, r in results.items():
        print(f"{name:<14}{r['n_features_input']:>6}"
              f"{r['holdout']['scorecard']['AUC']:>15.4f}{r['holdout']['xgboost']['AUC']:>13.4f}")
    print(f"\nSaved {out} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
