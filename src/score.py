"""
Single entry point for scoring an application.

    from score import load_model, score_application
    model = load_model()
    result = score_application(model, {"AMT_CREDIT": 250000, ...})

Inputs use the micro-profile field names (see micro_profile.FEATURE_SPEC).
Application basics are required; bureau and lender-history fields are optional
and treated as missing (thin file) when absent. The same `build_micro_features`
function used in training prepares the inputs, so there is no train/serve skew.
"""

from __future__ import annotations

import pathlib

import joblib
import pandas as pd

from decision import PolicyConfig, decide, policy_checks
from micro_profile import BUREAU_SCORE_PROXY, FEATURE_SPEC, build_micro_features

MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "micro_scorecard.pkl"
LABELS = {name: meaning for name, _, meaning, _ in FEATURE_SPEC}
LABELS.update({c: f"Bureau score proxy {i}" for i, c in enumerate(BUREAU_SCORE_PROXY, 1)})
REQUIRED = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS",
            "AGE_YEARS", "EMPLOYMENT_YEARS"]


def load_model(path: pathlib.Path = MODEL_PATH) -> dict:
    """Artefact = {"scorecard": MicroScorecard, "include_bureau_score": bool, ...}."""
    return joblib.load(path)


def score_application(model: dict, application: dict, policy: PolicyConfig = PolicyConfig()) -> dict:
    missing = [k for k in REQUIRED if application.get(k) is None]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    raw = pd.DataFrame([application])
    X = build_micro_features(raw, include_bureau_score=model["include_bureau_score"])
    card = model["scorecard"]

    score = int(card.score(X).iloc[0])
    band = str(card.band([score]).iloc[0])
    row = build_micro_features(raw, include_bureau_score=True).iloc[0]
    fired = policy_checks(row, policy)
    adverse, positive = card.reasons(X, LABELS)

    return {
        "decision": decide(band, fired),
        "score": score,
        "band": band,
        "default_probability": round(float(card.pd_hat(X)[0]), 4),
        "adverse_reasons": adverse,
        "positive_reasons": positive,
        "policy_flags": fired,
        "thin_file": bool(X["HAS_BUREAU_FILE"].iloc[0] == 0),
    }
