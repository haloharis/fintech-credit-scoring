"""Tests for the micro scorecard, decision engine and scoring entry point."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from decision import PolicyConfig, decide, policy_checks  # noqa: E402
from micro_profile import build_micro_features  # noqa: E402
from scorecard import BAND_NAMES, MicroScorecard  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    """Synthetic data where x_good lowers risk and x_bad raises it."""
    rng = np.random.default_rng(0)
    n = 20_000
    x_good, x_bad, noise = rng.normal(size=(3, n))
    p = 1 / (1 + np.exp(2.5 - 1.0 * x_bad + 1.0 * x_good))  # P(default)
    y = pd.Series((rng.random(n) < p).astype(int))
    X = pd.DataFrame({"x_good": x_good, "x_bad": x_bad, "noise": noise})
    card = MicroScorecard(top_n=3, iv_min=0.1, iv_max=5.0).fit(X, y)
    card.set_bands(card.score(X))
    return card, X, y


def test_higher_score_means_lower_risk(fitted):
    card, X, y = fitted
    scores = card.score(X)
    assert scores[y == 0].mean() > scores[y == 1].mean()


def test_all_coefficients_have_expected_sign(fitted):
    card, _, _ = fitted
    assert (card.lr.coef_[0] <= 0).all()


def test_contributions_sum_to_score(fitted):
    card, X, _ = fitted
    total = card.contributions(X).sum(axis=1).round().astype(int)
    assert (total == card.score(X)).all()


def test_points_follow_woe_direction(fitted):
    """Within a feature, bins with higher WoE (safer) must earn more points."""
    card, _, _ = fitted
    tbl = card.points_table()
    for _, g in tbl.groupby("feature"):
        assert g["woe"].corr(g["points"]) > 0.99


def test_bad_rate_falls_across_bands(fitted):
    card, X, y = fitted
    bands = card.band(card.score(X))
    rates = pd.DataFrame({"b": bands.values, "y": y.values}).groupby("b")["y"].mean().reindex(BAND_NAMES)
    assert rates.is_monotonic_decreasing


def test_reasons_point_at_the_weak_feature(fitted):
    card, X, _ = fitted
    row = X.iloc[[0]].copy()
    row["x_bad"] = 3.0  # clearly risky
    adverse, _ = card.reasons(row, {})
    assert adverse and adverse[0][0] == "x_bad"


# ── Decision engine ───────────────────────────────────────────
def test_knockout_overrides_good_band():
    fired = policy_checks(pd.Series({"AGE_YEARS": 35, "max_bureau_day_overdue": 120}))
    assert decide("Excellent", fired) == "Decline"


def test_refer_rule_downgrades_approve_only():
    fired = policy_checks(pd.Series({"AGE_YEARS": 35, "ANNUITY_INCOME_RATIO": 0.7}))
    assert decide("Good", fired) == "Refer"
    assert decide("Very Poor", fired) == "Decline"


def test_clean_applicant_follows_band():
    fired = policy_checks(pd.Series({"AGE_YEARS": 35, "ANNUITY_INCOME_RATIO": 0.1}), PolicyConfig())
    assert fired == []
    assert decide("Fair", fired) == "Refer"


# ── Feature parity (training table vs. runtime path) ─────────
TABLE = ROOT / "data" / "processed" / "train_features.csv"


@pytest.mark.skipif(not TABLE.exists(), reason="processed feature table not built")
def test_runtime_features_match_training_table():
    df = pd.read_csv(TABLE, nrows=5000)
    rebuilt = build_micro_features(df)
    for col in ["LOAN_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
                "INCOME_PER_PERSON", "AGE_YEARS", "EMPLOYMENT_YEARS", "EMPLOY_TO_AGE_RATIO"]:
        np.testing.assert_allclose(rebuilt[col], df[col], equal_nan=True, err_msg=col)


def test_missing_bureau_fields_become_thin_file():
    app = pd.DataFrame([{"AMT_CREDIT": 100000, "AMT_ANNUITY": 8000, "AMT_INCOME_TOTAL": 90000,
                         "CNT_FAM_MEMBERS": 3, "AGE_YEARS": 30, "EMPLOYMENT_YEARS": 4}])
    X = build_micro_features(app)
    assert X["HAS_BUREAU_FILE"].iloc[0] == 0 and X["HAS_LENDER_HISTORY"].iloc[0] == 0
    assert X.shape[1] == 41
