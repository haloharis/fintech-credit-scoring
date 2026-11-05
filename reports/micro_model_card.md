# Model Card — Micro-lending Scorecard (benchmark build)

**Status:** demonstration model trained on public benchmark data. Not validated on, or suitable for, live lending decisions at any institution until retrained on that institution's data.

## Purpose
Rank applicants for micro-enterprise loans by likelihood of default, and turn the ranking into an
Approve / Refer / Decline recommendation with reasons. Intended as the first product profile of a
scoring platform that later extends to SME, gold and vehicle lending.

## Data
- **Source:** Home Credit Default Risk (Kaggle), 307,511 applications, 8.07% default rate.
- **Proxy mapping:** benchmark columns stand in for the data a microfinance bank would hold
  (bureau history, repayment history with the lender, application details). The full mapping, with the
  intended LOLC source for each field, is in `src/micro_profile.py` and shown in the dashboard.
- **Split:** stratified random 80/20 (seed 42). 246,008 train / 61,503 hold-out. All encoders, feature
  selection, scaling and models are fit on the train split only; the hold-out is scored once.
- **Not in the data:** declined applicants (no reject inference), business cash flow, loan-officer
  assessment, collateral, group lending behaviour, Pakistani market context.

## Model
- **Champion:** WoE-binned logistic regression, 21 selected features (IV 0.02–0.50, pairwise
  correlation < 0.8, coefficient signs enforced). Score = 600 at 30:1 good-to-bad odds, 20 points to double
  the odds. Higher score = lower risk. Five equal-population bands calibrated on the training scores.
- **Challenger:** XGBoost on the same 41 inputs (comparison only; not used for decisions).
- **Excluded inputs:** gender, family status, and engineered polynomial / hand-weighted composite features.
- **Decision layer:** policy rules above the model (minimum age; decline on 90+ days overdue at the bureau;
  refer on instalment > 50% of income or 3+ enquiries in a quarter). Thresholds are placeholders.

## Performance (hold-out, 61,503 applicants)
| Model | AUC | Gini | KS |
|---|---|---|---|
| Scorecard (champion) | 0.747 | 0.493 | 0.369 |
| XGBoost (challenger) | 0.770 | 0.541 | 0.407 |

| Band | Share | Default rate |
|---|---|---|
| Very Poor | 18.9% | 20.5% |
| Poor | 20.5% | 9.7% |
| Fair | 20.0% | 5.6% |
| Good | 19.9% | 3.5% |
| Excellent | 20.7% | 1.9% |

| Decision | Share | Default rate |
|---|---|---|
| Approve | 39.8% | 2.7% |
| Refer | 41.1% | 7.6% |
| Decline | 19.2% | 20.3% |

Feature-set sensitivity (hold-out AUC, scorecard / XGBoost): full 298 features 0.755 / 0.790;
micro profile 0.748 / 0.770; micro profile without the three anonymised external scores 0.694 / 0.726.
Performance depends heavily on having a strong bureau-style score.

## Limitations
- Different country, product and population from the target lender; figures describe the method only.
- Approved-applicant sample, so declined cases are absent and decision volumes would differ in practice.
- Random split: Home Credit has no application date, so performance over time was not tested.
- Missing values are scored neutrally; the thin-file flags carry that risk and may need a separate segment.
- Out-of-range inputs take the nearest training bin.
- Policy thresholds and band cut-offs are assumptions pending the lender's credit policy and risk appetite.

## Fairness
Gender and family status are not model inputs. Subgroup performance has not been audited on this build;
a fairness review (by age band, region and, where lawful, gender) is part of onboarding.

## Monitoring plan
Monthly PSI on the score and key inputs (< 0.10 stable, 0.10–0.25 watch, > 0.25 review), band mix,
approval rate and realised default rate by band. The dashboard demonstrates this on simulated shifts.

## Reproduce
```
python src/feature_engineering.py --input raw_data/home-credit-default-risk --output data/processed
python scripts/baseline.py            # full-feature baseline
python scripts/micro_experiment.py    # feature-set comparison
python scripts/train_micro.py         # scorecard + decision evaluation
python scripts/build_demo_assets.py   # challenger + dashboard data
streamlit run app/dashboard.py
```
