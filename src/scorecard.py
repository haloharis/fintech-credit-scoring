"""
Points-based credit scorecard built on WoE + logistic regression.

Conventions (kept explicit because they are easy to get backwards):
  * WoE = ln(good / bad), so a higher WoE means lower risk.
  * The logistic model predicts P(bad) from WoE values, so every coefficient
    should be NEGATIVE. `fit` enforces this by dropping any feature whose
    coefficient comes out positive (a sign flip signals collinearity).
  * Score = offset + factor * ln(good / bad). HIGHER SCORE = LOWER RISK.
  * `base_odds` is good:bad odds at `base_score`; `pdo` points double the odds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from woe_encoder import WoEEncoder

BAND_NAMES = ["Very Poor", "Poor", "Fair", "Good", "Excellent"]  # ascending score


@dataclass
class MicroScorecard:
    bins: int = 10
    iv_min: float = 0.02
    iv_max: float = 0.50
    top_n: int = 25
    corr_max: float = 0.80
    C: float = 1.0
    pdo: float = 20.0
    base_score: float = 600.0
    base_odds: float = 30.0  # good:bad
    seed: int = 42

    encoder: WoEEncoder | None = field(default=None, repr=False)
    cols: list[str] = field(default_factory=list)
    lr: LogisticRegression | None = field(default=None, repr=False)
    ref_points: pd.Series | None = field(default=None, repr=False)
    band_cutoffs: list[float] = field(default_factory=list)

    # ── scale ─────────────────────────────────────────────────
    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    # ── fit ───────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y: pd.Series) -> MicroScorecard:
        self.encoder = WoEEncoder(bins=self.bins).fit(X, y)
        iv = self.encoder.get_iv_summary()["IV"]
        cands = iv[(iv >= self.iv_min) & (iv <= self.iv_max)].index.tolist()

        woe = self.encoder.transform(X[cands]).fillna(0)
        corr = woe.corr().abs()
        selected: list[str] = []
        for col in cands:  # strongest IV first
            if all(corr.loc[col, s] < self.corr_max for s in selected):
                selected.append(col)
            if len(selected) == self.top_n:
                break

        # Refit, dropping any feature with a wrong-sign (positive) coefficient.
        while True:
            lr = LogisticRegression(C=self.C, max_iter=1000, random_state=self.seed)
            lr.fit(woe[selected], y)
            wrong = [(c, f) for c, f in zip(lr.coef_[0], selected) if c > 0]
            if not wrong:
                break
            selected.remove(max(wrong)[1])
        self.cols, self.lr = selected, lr

        pts = self.contributions(X)
        self.ref_points = pts.mean()
        return self

    # ── scoring ───────────────────────────────────────────────
    def _woe(self, X: pd.DataFrame) -> pd.DataFrame:
        # Values beyond the training range take the nearest edge bin (an extreme
        # applicant must not look neutral). Truly missing values map to WoE 0;
        # the thin-file flags carry that risk.
        Xc = X[self.cols].copy()
        for col in self.cols:
            edges = self.encoder.bin_edges.get(col)
            if edges is not None:
                Xc[col] = Xc[col].clip(lower=edges[0], upper=edges[-1])
        return self.encoder.transform(Xc).fillna(0)

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature points; each row sums to the (unrounded) score."""
        woe = self._woe(X)
        coef = self.lr.coef_[0]
        base = (self.offset - self.factor * self.lr.intercept_[0]) / len(self.cols)
        return pd.DataFrame(base - self.factor * woe.values * coef, columns=self.cols, index=X.index)

    def score(self, X: pd.DataFrame) -> pd.Series:
        return self.contributions(X).sum(axis=1).round().astype(int)

    def pd_hat(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self._woe(X))[:, 1]

    # ── bands ─────────────────────────────────────────────────
    def set_bands(self, train_scores: pd.Series) -> None:
        """Five bands with equal share of the TRAINING population."""
        self.band_cutoffs = [float(c) for c in np.quantile(train_scores, [0.2, 0.4, 0.6, 0.8])]

    def band(self, scores) -> pd.Series:
        idx = np.searchsorted(self.band_cutoffs, np.asarray(scores), side="right")
        index = scores.index if isinstance(scores, pd.Series) else None
        return pd.Series([BAND_NAMES[i] for i in idx], index=index)

    # ── explanation ───────────────────────────────────────────
    def reasons(self, X: pd.DataFrame, labels: dict[str, str], n_adverse: int = 3, n_positive: int = 2,
                min_points: float = 1.0):
        """Reason codes for one applicant (a 1-row frame): points vs. the population average."""
        diff = (self.contributions(X).iloc[0] - self.ref_points).sort_values()
        adverse = [(labels.get(f, f), round(float(v), 1)) for f, v in diff.head(n_adverse).items() if v <= -min_points]
        positive = [(labels.get(f, f), round(float(v), 1)) for f, v in diff.tail(n_positive)[::-1].items() if v >= min_points]
        return adverse, positive

    # ── tables ────────────────────────────────────────────────
    def points_table(self) -> pd.DataFrame:
        """Points per bin for every scorecard feature (the classic scorecard layout)."""
        coef = dict(zip(self.cols, self.lr.coef_[0]))
        base = (self.offset - self.factor * self.lr.intercept_[0]) / len(self.cols)
        rows = []
        for f in self.cols:
            for bin_label, woe in self.encoder.woe_dict[f].items():
                rows.append({"feature": f, "bin": str(bin_label), "woe": round(woe, 4),
                             "points": round(base - self.factor * coef[f] * woe, 1)})
        return pd.DataFrame(rows)
