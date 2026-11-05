"""
Honest baseline — scorecard (WoE + logistic) vs XGBoost on a clean split.

Everything that learns from the target (WoE bins, IV selection, scaling,
model weights, early stopping) is fit on the TRAIN portion only. The hold-out
is scored exactly once, at the end.

Usage:
    python scripts/baseline.py --data data/processed --out reports/baseline_metrics.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import compute_gini, compute_ks_statistic  # noqa: E402
from woe_encoder import WoEEncoder  # noqa: E402

SEED = 42
N_FOLDS = 5
TEST_SIZE = 0.20
IV_MIN, IV_MAX, TOP_N, CORR_MAX = 0.02, 0.50, 40, 0.80


def metrics(y_true, y_prob) -> dict:
    return {
        "AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "Gini": round(float(compute_gini(y_true, y_prob)), 4),
        "KS": round(float(compute_ks_statistic(y_true, y_prob)), 4),
    }


# ── Scorecard ─────────────────────────────────────────────────
def fit_scorecard(X: pd.DataFrame, y: pd.Series) -> dict:
    enc = WoEEncoder(bins=10).fit(X, y)
    iv = enc.get_iv_summary()["IV"]
    cands = iv[(iv >= IV_MIN) & (iv <= IV_MAX)].head(TOP_N * 2).index.tolist()

    woe = enc.transform(X[cands]).fillna(0)
    corr = woe.corr().abs()
    selected: list[str] = []
    for col in cands:  # already ordered by IV, strongest first
        if all(corr.loc[col, s] < CORR_MAX for s in selected):
            selected.append(col)
        if len(selected) == TOP_N:
            break

    scaler = StandardScaler().fit(woe[selected])
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
    lr.fit(scaler.transform(woe[selected]), y)
    return {"enc": enc, "cols": selected, "scaler": scaler, "lr": lr}


def predict_scorecard(m: dict, X: pd.DataFrame) -> np.ndarray:
    woe = m["enc"].transform(X[m["cols"]]).fillna(0)
    return m["lr"].predict_proba(m["scaler"].transform(woe))[:, 1]


# ── XGBoost ───────────────────────────────────────────────────
def xgb_params(y: pd.Series) -> dict:
    return {
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "reg_lambda": 5.0,
        "scale_pos_weight": float((y == 0).sum() / (y == 1).sum()),
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": SEED,
        "n_jobs": -1,
    }


def run_models(X: pd.DataFrame, y: pd.Series, t0: float | None = None) -> dict:
    """Split, cross-validate on train, score the hold-out once. Returns a report dict."""
    t0 = t0 or time.time()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_tr, y_tr = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
    print(f"Train {len(X_tr):,} | Hold-out {len(X_te):,} | {X.shape[1]} features", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv = {"scorecard": [], "xgboost": []}
    best_iters: list[int] = []

    for fold, (tr, va) in enumerate(skf.split(X_tr, y_tr), 1):
        Xa, ya, Xv, yv = X_tr.iloc[tr], y_tr.iloc[tr], X_tr.iloc[va], y_tr.iloc[va]

        sc = fit_scorecard(Xa, ya)
        cv["scorecard"].append(metrics(yv, predict_scorecard(sc, Xv)))

        mdl = xgb.XGBClassifier(
            **xgb_params(ya), n_estimators=2000, early_stopping_rounds=50
        )
        mdl.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        best_iters.append(int(mdl.best_iteration) + 1)
        cv["xgboost"].append(metrics(yv, mdl.predict_proba(Xv)[:, 1]))

        print(
            f"fold {fold}: scorecard AUC {cv['scorecard'][-1]['AUC']:.4f} | "
            f"xgboost AUC {cv['xgboost'][-1]['AUC']:.4f} (iters {best_iters[-1]}) "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

    # Final fit on the full TRAIN portion; hold-out scored once.
    sc_final = fit_scorecard(X_tr, y_tr)
    n_est = int(np.mean(best_iters))
    xgb_final = xgb.XGBClassifier(**xgb_params(y_tr), n_estimators=n_est)
    xgb_final.fit(X_tr, y_tr, verbose=False)

    holdout = {
        "scorecard": metrics(y_te, predict_scorecard(sc_final, X_te)),
        "xgboost": metrics(y_te, xgb_final.predict_proba(X_te)[:, 1]),
    }

    def summarise(rows: list[dict]) -> dict:
        out = {}
        for k in ("AUC", "Gini", "KS"):
            vals = [r[k] for r in rows]
            out[k] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
        return out

    return {
        "split": {"type": "stratified random", "test_size": TEST_SIZE, "seed": SEED,
                  "n_train": len(X_tr), "n_holdout": len(X_te)},
        "n_features_input": X.shape[1],
        "scorecard_features": sc_final["cols"],
        "xgboost_n_estimators": n_est,
        "cv": {k: summarise(v) for k, v in cv.items()},
        "holdout": holdout,
    }


def print_summary(report: dict) -> None:
    print(f"{'Model':<12}{'CV AUC':>16}{'Hold-out AUC':>14}{'Gini':>8}{'KS':>8}")
    for name in ("scorecard", "xgboost"):
        c = report["cv"][name]["AUC"]
        h = report["holdout"][name]
        print(f"{name:<12}{c['mean']:>10.4f}±{c['std']:.4f}{h['AUC']:>14.4f}{h['Gini']:>8.4f}{h['KS']:>8.4f}")


def main(data_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    t0 = time.time()
    df = pd.read_csv(data_dir / "train_features.csv")
    y = df["TARGET"]
    X = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    print(f"Data: {X.shape[0]:,} rows | default rate {y.mean():.2%}", flush=True)

    report = run_models(X, y, t0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 60)
    print_summary(report)
    print(f"\nSaved {out_path}  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="reports/baseline_metrics.json")
    a = ap.parse_args()
    main(pathlib.Path(a.data), pathlib.Path(a.out))
