"""
Train the micro-lending scorecard and evaluate the full decision flow.

Everything is fit on the 80% train split (same split as baseline.py); the 20%
hold-out is used once to report discrimination, band bad-rates and decisions.

Usage:
    python scripts/train_micro.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from decision import PolicyConfig, decide, policy_checks  # noqa: E402
from evaluate import compute_gini, compute_ks_statistic  # noqa: E402
from micro_profile import build_micro_features  # noqa: E402
from scorecard import BAND_NAMES, MicroScorecard  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

SEED, TEST_SIZE = 42, 0.20
INCLUDE_BUREAU_SCORE = True


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "train_features.csv")
    y = df["TARGET"]
    X = build_micro_features(df, include_bureau_score=INCLUDE_BUREAU_SCORE)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    card = MicroScorecard().fit(X_tr, y_tr)
    tr_scores = card.score(X_tr)
    card.set_bands(tr_scores)
    print(f"Scorecard: {len(card.cols)} features | band cut-offs {[round(c) for c in card.band_cutoffs]}")

    # ── Hold-out discrimination (score is higher = safer, so negate for AUC of bad) ──
    te_scores = card.score(X_te)
    risk = -te_scores
    disc = {
        "AUC": round(float(roc_auc_score(y_te, risk)), 4),
        "Gini": round(float(compute_gini(y_te, risk)), 4),
        "KS": round(float(compute_ks_statistic(y_te, risk)), 4),
    }
    print("Hold-out:", disc)

    # ── Bands ──
    bands = card.band(te_scores)
    band_tbl = (
        pd.DataFrame({"band": bands.values, "bad": y_te.values})
        .groupby("band")["bad"].agg(n="count", bad_rate="mean")
        .reindex(BAND_NAMES)
    )
    band_tbl["share"] = band_tbl["n"] / band_tbl["n"].sum()
    monotonic = bool(band_tbl["bad_rate"].is_monotonic_decreasing)
    print("\nBand       share  bad rate")
    for b, r in band_tbl.iterrows():
        print(f"{b:<10}{r['share']:>6.1%}{r['bad_rate']:>9.2%}")
    print("Bad rate falls monotonically with score:", monotonic)

    # ── Full decision flow on the hold-out ──
    cfg = PolicyConfig()
    decisions = []
    for (_, row), band in zip(X_te.iterrows(), bands):
        decisions.append(decide(band, policy_checks(row, cfg)))
    dec_tbl = (
        pd.DataFrame({"decision": decisions, "bad": y_te.values})
        .groupby("decision")["bad"].agg(n="count", bad_rate="mean")
    )
    dec_tbl["share"] = dec_tbl["n"] / dec_tbl["n"].sum()
    print("\nDecision   share  bad rate")
    for d in ("Approve", "Refer", "Decline"):
        if d in dec_tbl.index:
            r = dec_tbl.loc[d]
            print(f"{d:<10}{r['share']:>6.1%}{r['bad_rate']:>9.2%}")

    # ── Save ──
    model = {"scorecard": card, "include_bureau_score": INCLUDE_BUREAU_SCORE,
             "trained_on": f"{len(X_tr):,} benchmark rows (Home Credit, proxy data)"}
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "micro_scorecard.pkl")

    report = {
        "n_features": len(card.cols), "features": card.cols,
        "scale": {"base_score": card.base_score, "base_odds_good_to_bad": card.base_odds, "pdo": card.pdo},
        "band_cutoffs": card.band_cutoffs,
        "holdout": disc,
        "bands": band_tbl.reset_index().round(4).to_dict("records"),
        "bad_rate_monotonic": monotonic,
        "decisions": dec_tbl.reset_index().round(4).to_dict("records"),
        "points_table": card.points_table().to_dict("records"),
    }
    (ROOT / "reports" / "micro_scorecard_report.json").write_text(json.dumps(report, indent=2))
    print("\nSaved models/micro_scorecard.pkl and reports/micro_scorecard_report.json")


if __name__ == "__main__":
    main()
