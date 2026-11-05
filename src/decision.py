"""
Decision engine — turns a score and an applicant's data into
Approve / Refer / Decline, with the reasons that drove it.

Policy rules sit ABOVE the model, as in a real lender: a hard knock-out wins
over a good score. Thresholds are assumptions for the benchmark build and are
meant to be replaced with LOLC's credit policy at onboarding.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Band -> default action when no policy rule overrides it.
BAND_ACTION = {
    "Excellent": "Approve",
    "Good": "Approve",
    "Fair": "Refer",
    "Poor": "Refer",
    "Very Poor": "Decline",
}


@dataclass(frozen=True)
class PolicyConfig:
    min_age: float = 21.0
    severe_overdue_days: float = 90.0     # bureau worst days overdue -> decline
    max_annuity_to_income: float = 0.50   # affordability ceiling -> refer
    max_enquiries_quarter: float = 3.0    # credit-hunger signal -> refer


def policy_checks(row: pd.Series, cfg: PolicyConfig = PolicyConfig()) -> list[dict]:
    """Return the policy rules that fired as [{rule, action, detail}]."""
    fired: list[dict] = []

    def val(name: str) -> float:
        v = row.get(name)
        return float("nan") if v is None else float(v)

    if val("AGE_YEARS") < cfg.min_age:
        fired.append({"rule": "minimum_age", "action": "Decline",
                      "detail": f"Applicant under {cfg.min_age:.0f}"})
    if val("max_bureau_day_overdue") >= cfg.severe_overdue_days:
        fired.append({"rule": "severe_bureau_delinquency", "action": "Decline",
                      "detail": f"{cfg.severe_overdue_days:.0f}+ days overdue on a bureau facility"})
    if val("ANNUITY_INCOME_RATIO") > cfg.max_annuity_to_income:
        fired.append({"rule": "affordability", "action": "Refer",
                      "detail": f"Instalment above {cfg.max_annuity_to_income:.0%} of income"})
    if val("AMT_REQ_CREDIT_BUREAU_QRT") >= cfg.max_enquiries_quarter:
        fired.append({"rule": "recent_enquiries", "action": "Refer",
                      "detail": f"{cfg.max_enquiries_quarter:.0f}+ bureau enquiries in the last quarter"})
    return fired


def decide(band: str, fired: list[dict]) -> str:
    """Decline beats Refer beats the band's default action."""
    actions = {f["action"] for f in fired}
    if "Decline" in actions:
        return "Decline"
    action = BAND_ACTION[band]
    if "Refer" in actions and action == "Approve":
        return "Refer"
    return action
