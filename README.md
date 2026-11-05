# Credit Scoring MVP — Micro-lending

A credit scoring platform prototype for a microfinance lender. It turns an application, bureau history and
repayment history into a **score, a band, an Approve / Refer / Decline recommendation, and the reasons behind it**,
with monitoring for population drift.

> **Status: demonstration build.** It is trained on a public benchmark (Home Credit Default Risk), not on any
> lender's data. The figures show that the method works; real performance is measured after retraining on the
> lender's own portfolio. See [reports/micro_model_card.md](reports/micro_model_card.md).

## What's in it
| Piece | Where |
|---|---|
| Micro-lending feature profile, with the source of each field at the lender | [src/micro_profile.py](src/micro_profile.py) |
| Points-based scorecard (WoE + logistic), bands, reason codes | [src/scorecard.py](src/scorecard.py) |
| Decision engine (policy rules above the model) | [src/decision.py](src/decision.py) |
| One entry point, `score_application()`, shared with training | [src/score.py](src/score.py) |
| Demo dashboard (score an applicant, performance, monitoring, roadmap) | [app/dashboard.py](app/dashboard.py) |
| Honest baseline and feature-set experiments | [scripts/](scripts/) |
| Model card | [reports/micro_model_card.md](reports/micro_model_card.md) |

## Results (hold-out of 61,503 applicants, never used for training)
| Model | AUC | Gini | KS |
|---|---|---|---|
| Scorecard (champion) | 0.747 | 0.493 | 0.369 |
| XGBoost (challenger) | 0.770 | 0.541 | 0.407 |

Default rate falls from 20.5% in the lowest score band to 1.9% in the highest. Without the three anonymised
external scores (the stand-in for a bureau score) the scorecard drops to AUC 0.69, so bureau data quality will
drive real-world results.

## Run it
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# macOS only: brew install libomp   (needed by XGBoost)

# Raw Kaggle tables go in raw_data/home-credit-default-risk/ (not committed)
.venv/bin/python src/feature_engineering.py --input raw_data/home-credit-default-risk --output data/processed
.venv/bin/python scripts/train_micro.py
.venv/bin/python scripts/build_demo_assets.py
.venv/bin/streamlit run app/dashboard.py
.venv/bin/python -m pytest tests
```

## Design notes
- **One feature path.** Training and scoring both call `build_micro_features`; a test checks that runtime
  features match the training table exactly.
- **Honest evaluation.** Everything that learns from the target is fit on the training split only; the hold-out
  is scored once.
- **Higher score = lower risk.** 600 points at 30:1 good-to-bad odds, 20 points to double the odds. Coefficient
  signs are enforced during fitting.
- **Protected attributes excluded.** Gender and family status are not model inputs.
- **Policy above the model.** Hard rules override a good score, as in a real lender. Thresholds are placeholders.

## Legacy components
`api/`, `monitoring/`, `notebooks/`, `sql/` and the older models in `models/` come from the upstream project and
predate this build. The API and the older models still run but are not wired to the new scorecard.

## Credits
Built on top of [erikim-dev/credit-scorecard-pipeline](https://github.com/erikim-dev/credit-scorecard-pipeline) by
Eric Kimutai, whose feature pipeline, WoE encoder and evaluation utilities are reused here. Data: Home Credit
Default Risk (Kaggle), used under the competition's terms.

## Licence
MIT, see [LICENSE](LICENSE).
