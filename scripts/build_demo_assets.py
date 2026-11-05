"""
Build everything the demo dashboard needs, so the app never loads the 600 MB table.

  * trains the XGBoost challenger on the same micro features and train split
  * hold-out performance for champion vs challenger (metrics, ROC, score histogram)
  * drift scenarios: the hold-out is perturbed in documented ways and compared with
    the training population (PSI on the score and on key features)

Outputs: models/micro_xgb_challenger.pkl, reports/demo_assets.json

Usage:
    python scripts/train_micro.py && python scripts/build_demo_assets.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import compute_gini, compute_ks_statistic, compute_psi  # noqa: E402
from micro_profile import build_micro_features  # noqa: E402
from scorecard import BAND_NAMES  # noqa: E402

SEED, TEST_SIZE = 42, 0.20


def metrics(y, p) -> dict:
    return {"AUC": round(float(roc_auc_score(y, p)), 4),
            "Gini": round(float(compute_gini(y, p)), 4),
            "KS": round(float(compute_ks_statistic(y, p)), 4)}


def roc_points(y, p, n: int = 80) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.linspace(0, len(fpr) - 1, n).astype(int)
    return {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}


# Each scenario edits hold-out features the way a real population shift would.
def scenario_frames(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    out["Stable (no change)"] = X.copy()

    mild = X.copy()
    for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        mild[c] = (mild[c] * 0.88).clip(0, 1)
    mild["debt_to_credit_ratio"] = (mild["debt_to_credit_ratio"] + 0.15).clip(upper=1.5)
    out["Mild shift - tighter household budgets"] = mild

    severe = X.copy()
    for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        severe[c] = (severe[c] * 0.80).clip(0, 1)
    severe["debt_to_credit_ratio"] = (severe["debt_to_credit_ratio"] + 0.25).clip(upper=1.5)
    severe["late_payment_ratio"] = (severe["late_payment_ratio"] + 0.10).clip(upper=1)
    severe["AMT_CREDIT"] = severe["AMT_CREDIT"] * 1.30
    out["Severe shift - income shock"] = severe
    return out


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "train_features.csv")
    y = df["TARGET"]
    model = joblib.load(ROOT / "models" / "micro_scorecard.pkl")
    card = model["scorecard"]
    X = build_micro_features(df, include_bureau_score=model["include_bureau_score"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    # ── Challenger: same features, same split; n_estimators from the Phase 1 CV ──
    base = json.loads((ROOT / "reports" / "micro_experiment.json").read_text())["micro"]
    n_est = int(base["xgboost_n_estimators"])
    challenger = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, reg_lambda=5.0, n_estimators=n_est, tree_method="hist",
        scale_pos_weight=float((y_tr == 0).sum() / (y_tr == 1).sum()),
        random_state=SEED, n_jobs=-1,
    ).fit(X_tr, y_tr)
    joblib.dump({"model": challenger, "features": list(X.columns)}, ROOT / "models" / "micro_xgb_challenger.pkl")

    sc_te = card.score(X_te)
    risk_sc = -sc_te.values
    p_xgb = challenger.predict_proba(X_te)[:, 1]

    # ── Score histogram by outcome (10-point bins) ──
    lo, hi = int(np.floor(sc_te.min() / 10) * 10), int(np.ceil(sc_te.max() / 10) * 10)
    edges = np.arange(lo, hi + 10, 10)
    good = np.histogram(sc_te[y_te.values == 0], bins=edges)[0]
    bad = np.histogram(sc_te[y_te.values == 1], bins=edges)[0]

    # ── Drift scenarios (population = training scores, actual = perturbed hold-out) ──
    tr_scores = card.score(X_tr)
    key_features = ["debt_to_credit_ratio", "EXT_SOURCE_2", "late_payment_ratio", "AMT_CREDIT",
                    "AMT_REQ_CREDIT_BUREAU_QRT", "AGE_YEARS"]
    drift = []
    for name, Xs in scenario_frames(X_te).items():
        sc = card.score(Xs)
        feat_psi = {}
        for f in key_features:
            a, e = Xs[f].dropna().values, X_tr[f].dropna().values
            feat_psi[f] = compute_psi(e, a)["PSI"]
        bands = card.band(sc).value_counts(normalize=True).reindex(BAND_NAMES).fillna(0)
        drift.append({
            "scenario": name,
            "score_psi": compute_psi(tr_scores.values, sc.values),
            "mean_score": round(float(sc.mean()), 1),
            "feature_psi": feat_psi,
            "band_mix": {b: round(float(v), 4) for b, v in bands.items()},
        })

    report = json.loads((ROOT / "reports" / "micro_scorecard_report.json").read_text())
    assets = {
        "metrics": {"Scorecard (champion)": metrics(y_te, risk_sc), "XGBoost (challenger)": metrics(y_te, p_xgb)},
        "roc": {"Scorecard (champion)": roc_points(y_te, risk_sc), "XGBoost (challenger)": roc_points(y_te, p_xgb)},
        "score_hist": {"edges": edges.tolist(), "good": good.tolist(), "bad": bad.tolist()},
        "train_band_mix": {b: 0.2 for b in BAND_NAMES},
        "bands": report["bands"],
        "decisions": report["decisions"],
        "band_cutoffs": report["band_cutoffs"],
        "scale": report["scale"],
        "n_holdout": int(len(y_te)),
        "holdout_default_rate": round(float(y_te.mean()), 4),
        "drift": drift,
    }
    (ROOT / "reports" / "demo_assets.json").write_text(json.dumps(assets, indent=2))
    print(json.dumps(assets["metrics"], indent=2))
    for d in drift:
        print(f"{d['scenario']:<42} score PSI {d['score_psi']['PSI']:.3f} ({d['score_psi']['Status']}) mean score {d['mean_score']}")
    print("Saved models/micro_xgb_challenger.pkl and reports/demo_assets.json")


if __name__ == "__main__":
    main()
