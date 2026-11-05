"""
Micro-lending product profile — the feature set a microfinance lender could
realistically supply at application time, and where each field would come from
at LOLC MFB once real data is available.

The Home Credit benchmark columns are stand-ins ("proxy") for those sources.
Gender, family status and other protected or proxy attributes are deliberately
excluded; they may be kept for fairness reporting only.

`build_micro_features` is the single place these features are defined, so
training and scoring cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (feature, group, proxy meaning, LOLC source at onboarding)
FEATURE_SPEC: list[tuple[str, str, str, str]] = [
    # Repayment capacity -------------------------------------------------
    ("AMT_CREDIT", "capacity", "Requested loan amount", "Application"),
    ("AMT_ANNUITY", "capacity", "Instalment amount", "Application / loan officer"),
    ("AMT_INCOME_TOTAL", "capacity", "Stated income", "Loan officer cash-flow assessment"),
    ("LOAN_INCOME_RATIO", "capacity", "Loan / income", "Derived"),
    ("ANNUITY_INCOME_RATIO", "capacity", "Instalment / income", "Derived"),
    ("CREDIT_TERM", "capacity", "Instalment / loan amount (inverse tenor)", "Derived"),
    ("INCOME_PER_PERSON", "capacity", "Income per household member", "Application"),
    ("CNT_FAM_MEMBERS", "capacity", "Household size", "Application"),
    # Stability ----------------------------------------------------------
    ("AGE_YEARS", "stability", "Applicant age", "CNIC / application"),
    ("EMPLOYMENT_YEARS", "stability", "Years in job (proxy: years in business)", "Loan officer"),
    ("EMPLOY_TO_AGE_RATIO", "stability", "Share of life employed", "Derived"),
    ("NO_EMPLOYMENT_RECORD", "stability", "Placeholder employment value (pensioner / none)", "Derived"),
    # Credit bureau ------------------------------------------------------
    ("bureau_loan_count", "bureau", "Number of bureau facilities", "Tasdeeq / eCIB"),
    ("active_credits", "bureau", "Open facilities", "Tasdeeq / eCIB"),
    ("total_debt", "bureau", "Outstanding bureau debt", "Tasdeeq / eCIB"),
    ("debt_to_credit_ratio", "bureau", "Outstanding / limit", "Tasdeeq / eCIB"),
    ("overdue_count", "bureau", "Facilities with overdue amount", "Tasdeeq / eCIB"),
    ("max_bureau_day_overdue", "bureau", "Worst days overdue", "Tasdeeq / eCIB 24-month grid"),
    ("bureau_loans_last_12m", "bureau", "New facilities in 12 months", "Tasdeeq / eCIB"),
    ("bureau_loans_last_24m", "bureau", "New facilities in 24 months", "Tasdeeq / eCIB"),
    ("earliest_bureau_days", "bureau", "Credit history length", "Tasdeeq / eCIB"),
    ("bb_dpd_ratio", "bureau", "Share of months past due", "Tasdeeq / eCIB 24-month grid"),
    ("bb_months_covered", "bureau", "Months of bureau history", "Tasdeeq / eCIB"),
    ("AMT_REQ_CREDIT_BUREAU_QRT", "bureau", "Enquiries in last quarter", "Tasdeeq / eCIB enquiries"),
    ("AMT_REQ_CREDIT_BUREAU_YEAR", "bureau", "Enquiries in last year", "Tasdeeq / eCIB enquiries"),
    # Repayment behaviour with the lender (relationship / loan cycle) ----
    ("late_payment_ratio", "behaviour", "Share of late instalments", "LOLC repayment history"),
    ("max_days_late", "behaviour", "Worst lateness", "LOLC repayment history"),
    ("late_ratio_last_12m", "behaviour", "Recent share of late instalments", "LOLC repayment history"),
    ("payment_to_due_ratio", "behaviour", "Paid / due", "LOLC repayment history"),
    ("total_installments", "behaviour", "Instalments observed (experience)", "LOLC repayment history"),
    ("max_dpd", "behaviour", "Worst days past due on prior loans", "LOLC repayment history"),
    ("contract_completion_rate", "behaviour", "Share of prior loans completed", "LOLC loan cycle"),
    # Application history ------------------------------------------------
    ("prev_app_count", "history", "Prior applications", "LOLC loan cycle"),
    ("approval_rate", "history", "Share approved", "LOLC loan cycle"),
    ("refused_apps", "history", "Prior refusals", "LOLC loan cycle"),
    ("refusals_last_12m", "history", "Refusals in 12 months", "LOLC loan cycle"),
    # Thin-file flags ----------------------------------------------------
    ("HAS_BUREAU_FILE", "thin_file", "Any bureau record", "Tasdeeq / eCIB"),
    ("HAS_LENDER_HISTORY", "thin_file", "Any prior loans with the lender", "LOLC"),
]

# Anonymised third-party scores in the benchmark. At LOLC the nearest analogue is a
# bureau score (Tasdeeq / eCIB), but its form and strength will differ, so the
# profile can be built with or without it.
BUREAU_SCORE_PROXY = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

FEATURE_NAMES = [name for name, *_ in FEATURE_SPEC]
DAYS_EMPLOYED_PLACEHOLDER = 365243  # Home Credit's code for "no employment record"


def build_micro_features(df: pd.DataFrame, include_bureau_score: bool = True) -> pd.DataFrame:
    """Return the micro-lending feature matrix from the engineered benchmark table."""
    out = df.copy()
    out["NO_EMPLOYMENT_RECORD"] = (out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_PLACEHOLDER).astype(int)
    out["HAS_BUREAU_FILE"] = out["bureau_loan_count"].notna().astype(int)
    out["HAS_LENDER_HISTORY"] = out["prev_app_count"].notna().astype(int)

    cols = list(FEATURE_NAMES)
    if include_bureau_score:
        cols += BUREAU_SCORE_PROXY
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns for micro profile: {missing}")
    return out[cols].replace([np.inf, -np.inf], np.nan)
