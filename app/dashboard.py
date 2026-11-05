"""
Credit scoring MVP — micro-lending demo dashboard.

Pages: Score an applicant | Model performance | Monitoring | Roadmap & onboarding
Run:   streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from micro_profile import FEATURE_SPEC, build_micro_features  # noqa: E402
from score import LABELS, load_model, score_application  # noqa: E402

# ── Palette (dark surface; categorical slots 1-2, fixed status colours) ───────
BLUE, ORANGE, GRID, MUTED = "#3987e5", "#d95926", "#2a2f36", "#8b949e"
STATUS = {  # always paired with an icon + label, never colour alone
    "Approve": ("#0ca30c", "✔"), "Refer": ("#fab219", "▲"), "Decline": ("#d03b3b", "✖"),
    "Stable": ("#0ca30c", "✔"), "Monitor": ("#fab219", "▲"), "Alert": ("#d03b3b", "✖"),
}
BAND_ORDER = ["Very Poor", "Poor", "Fair", "Good", "Excellent"]

st.set_page_config(page_title="Credit Scoring MVP", page_icon="📊", layout="wide")


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model() -> dict:
    return load_model()


@st.cache_resource
def get_challenger() -> dict:
    return joblib.load(ROOT / "models" / "micro_xgb_challenger.pkl")


@st.cache_data
def get_assets() -> dict:
    return json.loads((ROOT / "reports" / "demo_assets.json").read_text())


@st.cache_data
def get_report() -> dict:
    return json.loads((ROOT / "reports" / "micro_scorecard_report.json").read_text())


def style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12, x=0),
        font=dict(size=13),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def status_chip(label: str, kind: str) -> str:
    colour, icon = STATUS[kind]
    return (f"<span style='background:{colour}22;border:1px solid {colour};color:{colour};"
            f"padding:2px 10px;border-radius:12px;font-weight:600'>{icon} {label}</span>")


# ── Presets for the scoring form ──────────────────────────────────────────────
BASE = dict(loan=250_000, instalment=14_000, income=180_000, household=4, age=34, years_work=6,
            has_bureau=False, has_history=False, has_score=False,
            b_loans=2, b_active=1, b_debt=60_000, b_util=0.30, b_overdue_days=0, b_new12=1, b_new24=1,
            b_hist_years=4.0, b_dpd_ratio=0.0, b_enq_q=0, b_enq_y=1,
            h_prev=2, h_approval=1.0, h_refused=0, h_late=0.0, h_maxlate=0, h_instalments=24,
            s1=0.5, s2=0.5, s3=0.5)
PRESETS = {
    "New borrower (thin file)": dict(BASE),
    "Repeat borrower, clean record": {**BASE, "has_bureau": True, "has_history": True, "has_score": True,
                                      "b_loans": 3, "b_active": 1, "b_util": 0.20, "b_hist_years": 6.0,
                                      "h_prev": 4, "h_instalments": 60, "s1": 0.72, "s2": 0.75, "s3": 0.70},
    "Stretched borrower": {**BASE, "instalment": 24_000, "years_work": 1, "has_bureau": True, "has_history": True,
                           "has_score": True, "b_loans": 6, "b_active": 4, "b_util": 0.90, "b_new12": 3,
                           "b_new24": 4, "b_dpd_ratio": 0.25, "b_enq_q": 2, "h_approval": 0.3, "h_refused": 3,
                           "h_late": 0.4, "h_maxlate": 45, "s1": 0.20, "s2": 0.25, "s3": 0.20},
    "Severe overdue on bureau": {**BASE, "has_bureau": True, "has_score": True, "b_overdue_days": 120,
                                 "b_dpd_ratio": 0.4, "s1": 0.7, "s2": 0.8, "s3": 0.7},
}


def build_application(v: dict) -> dict:
    """Map form values to the micro-profile field names; sections left off stay missing."""
    app = {"AMT_CREDIT": v["loan"], "AMT_ANNUITY": v["instalment"], "AMT_INCOME_TOTAL": v["income"],
           "CNT_FAM_MEMBERS": v["household"], "AGE_YEARS": v["age"], "EMPLOYMENT_YEARS": v["years_work"]}
    if v["has_bureau"]:
        app.update(bureau_loan_count=v["b_loans"], active_credits=v["b_active"], total_debt=v["b_debt"],
                   debt_to_credit_ratio=v["b_util"], max_bureau_day_overdue=v["b_overdue_days"],
                   bureau_loans_last_12m=v["b_new12"], bureau_loans_last_24m=v["b_new24"],
                   earliest_bureau_days=-v["b_hist_years"] * 365.25, bb_dpd_ratio=v["b_dpd_ratio"],
                   bb_months_covered=v["b_hist_years"] * 12,
                   AMT_REQ_CREDIT_BUREAU_QRT=v["b_enq_q"], AMT_REQ_CREDIT_BUREAU_YEAR=v["b_enq_y"],
                   overdue_count=int(v["b_overdue_days"] > 0))
    if v["has_history"]:
        app.update(prev_app_count=v["h_prev"], approval_rate=v["h_approval"], refused_apps=v["h_refused"],
                   late_payment_ratio=v["h_late"], max_days_late=v["h_maxlate"],
                   total_installments=v["h_instalments"])
    if v["has_score"]:
        app.update(EXT_SOURCE_1=v["s1"], EXT_SOURCE_2=v["s2"], EXT_SOURCE_3=v["s3"])
    return app


# ── Page 1: score an applicant ───────────────────────────────────────────────
def page_score() -> None:
    st.header("Score an applicant")
    preset = st.selectbox("Start from an example", list(PRESETS))
    d = PRESETS[preset]
    k = lambda name: f"{preset}|{name}"  # noqa: E731  (fresh widgets when the example changes)

    left, right = st.columns([1, 1.15], gap="large")
    v: dict = {}
    with left:
        st.subheader("Application")
        c1, c2 = st.columns(2)
        v["loan"] = c1.number_input("Loan amount (PKR)", 10_000, 5_000_000, d["loan"], 10_000, key=k("loan"))
        v["instalment"] = c2.number_input("Instalment (PKR)", 500, 500_000, d["instalment"], 500, key=k("inst"))
        v["income"] = c1.number_input("Annual income (PKR)", 20_000, 20_000_000, d["income"], 10_000, key=k("inc"))
        v["household"] = c2.number_input("Household size", 1, 15, d["household"], key=k("hh"))
        v["age"] = c1.number_input("Age", 18, 80, d["age"], key=k("age"))
        v["years_work"] = c2.number_input("Years in business / job", 0.0, 50.0, float(d["years_work"]), 0.5, key=k("yw"))

        v["has_bureau"] = st.toggle("Credit bureau file available (Tasdeeq / eCIB)", d["has_bureau"], key=k("hb"))
        if v["has_bureau"]:
            with st.expander("Bureau details", expanded=True):
                a, b = st.columns(2)
                v["b_loans"] = a.number_input("Facilities (all time)", 0, 50, d["b_loans"], key=k("bl"))
                v["b_active"] = b.number_input("Open facilities", 0, 50, d["b_active"], key=k("ba"))
                v["b_debt"] = a.number_input("Outstanding debt (PKR)", 0, 20_000_000, d["b_debt"], 5_000, key=k("bd"))
                v["b_util"] = b.slider("Outstanding / limit", 0.0, 1.5, float(d["b_util"]), 0.05, key=k("bu"))
                v["b_overdue_days"] = a.number_input("Worst days overdue", 0, 365, d["b_overdue_days"], key=k("bo"))
                v["b_dpd_ratio"] = b.slider("Share of months past due", 0.0, 1.0, float(d["b_dpd_ratio"]), 0.05, key=k("bp"))
                v["b_new12"] = a.number_input("New facilities, 12 months", 0, 20, d["b_new12"], key=k("b12"))
                v["b_new24"] = b.number_input("New facilities, 24 months", 0, 30, d["b_new24"], key=k("b24"))
                v["b_hist_years"] = a.number_input("Credit history (years)", 0.0, 40.0, float(d["b_hist_years"]), 0.5, key=k("bh"))
                v["b_enq_q"] = b.number_input("Enquiries, last quarter", 0, 30, d["b_enq_q"], key=k("bq"))
                v["b_enq_y"] = a.number_input("Enquiries, last year", 0, 50, d["b_enq_y"], key=k("by"))
        v["has_history"] = st.toggle("Repeat borrower (history with the lender)", d["has_history"], key=k("hh2"))
        if v["has_history"]:
            with st.expander("Lender history", expanded=True):
                a, b = st.columns(2)
                v["h_prev"] = a.number_input("Previous applications", 0, 50, d["h_prev"], key=k("hp"))
                v["h_approval"] = b.slider("Approval rate", 0.0, 1.0, float(d["h_approval"]), 0.05, key=k("ha"))
                v["h_refused"] = a.number_input("Refused applications", 0, 50, d["h_refused"], key=k("hr"))
                v["h_late"] = b.slider("Share of late instalments", 0.0, 1.0, float(d["h_late"]), 0.05, key=k("hl"))
                v["h_maxlate"] = a.number_input("Worst days late", 0, 365, d["h_maxlate"], key=k("hm"))
                v["h_instalments"] = b.number_input("Instalments observed", 0, 500, d["h_instalments"], key=k("hi"))
        v["has_score"] = st.toggle("Bureau score available (benchmark proxy, 0-1)", d["has_score"], key=k("hs"))
        if v["has_score"]:
            a, b, c = st.columns(3)
            v["s1"] = a.slider("Score 1", 0.0, 1.0, float(d["s1"]), 0.01, key=k("s1"))
            v["s2"] = b.slider("Score 2", 0.0, 1.0, float(d["s2"]), 0.01, key=k("s2"))
            v["s3"] = c.slider("Score 3", 0.0, 1.0, float(d["s3"]), 0.01, key=k("s3"))

    app = build_application(v)
    model, challenger = get_model(), get_challenger()
    res = score_application(model, app)
    xrow = build_micro_features(pd.DataFrame([app]), include_bureau_score=model["include_bureau_score"])
    p_xgb = float(challenger["model"].predict_proba(xrow[challenger["features"]])[0, 1])

    with right:
        st.subheader("Decision")
        colour, icon = STATUS[res["decision"]]
        st.markdown(
            f"<div style='border:1px solid {colour};background:{colour}18;border-radius:10px;padding:14px 18px'>"
            f"<span style='font-size:1.9rem;font-weight:700;color:{colour}'>{icon} {res['decision']}</span>"
            f"<span style='margin-left:14px;color:{MUTED}'>Band: <b>{res['band']}</b></span></div>",
            unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Score", res["score"])
        m2.metric("Default risk (scorecard)", f"{res['default_probability']:.1%}")
        m3.metric("Default risk (XGBoost)", f"{p_xgb:.1%}")
        st.plotly_chart(score_bar(res["score"], model["scorecard"].band_cutoffs), width="stretch")

        if res["thin_file"]:
            st.info("Thin file: no bureau record. Missing history is treated neutrally and flagged for the credit officer.")
        for f in res["policy_flags"]:
            chip = status_chip(f["action"], f["action"])
            st.markdown(f"{chip} &nbsp; **{f['rule'].replace('_', ' ').title()}** — {f['detail']}", unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**What lowered the score**")
            for name, pts in res["adverse_reasons"] or [("Nothing material", 0)]:
                st.markdown(f"- {name} ({pts:+.1f} pts)" if pts else f"- {name}")
        with r2:
            st.markdown("**What helped**")
            for name, pts in res["positive_reasons"] or [("Nothing material", 0)]:
                st.markdown(f"- {name} ({pts:+.1f} pts)" if pts else f"- {name}")
        st.caption("Points are measured against the average applicant in the training data.")

    with st.expander("Where each input comes from at LOLC"):
        st.dataframe(pd.DataFrame(FEATURE_SPEC, columns=["Feature", "Group", "Benchmark meaning", "LOLC source at onboarding"]),
                     hide_index=True, width="stretch")


def score_bar(score: int, cutoffs: list[float]) -> go.Figure:
    edges = [min(score, cutoffs[0]) - 40] + cutoffs + [max(score, cutoffs[-1]) + 40]
    fig = go.Figure()
    for i, band in enumerate(BAND_ORDER):
        fig.add_trace(go.Bar(x=[edges[i + 1] - edges[i]], y=[""], base=edges[i], orientation="h", name=band,
                             marker=dict(color=BLUE, opacity=0.2 + 0.18 * i), hovertemplate=f"{band}<extra></extra>",
                             text=band, textposition="inside", insidetextanchor="middle"))
    fig.add_vline(x=score, line=dict(color="#f0efec", width=3))
    fig.update_layout(barmode="overlay", showlegend=False, xaxis_title="Score (higher = lower risk)")
    return style(fig, 130)


# ── Page 2: model performance ────────────────────────────────────────────────
def page_performance() -> None:
    a = get_assets()
    st.header("Model performance")
    st.caption(f"Hold-out of {a['n_holdout']:,} applicants never used for training. Default rate {a['holdout_default_rate']:.1%}.")
    st.warning("Trained on a public benchmark (Home Credit), not LOLC data. These figures show the method works; "
               "LOLC's own performance will be measured after retraining.")

    cols = st.columns(2)
    for col, (name, m) in zip(cols, a["metrics"].items()):
        with col:
            st.markdown(f"**{name}**")
            x, y, z = st.columns(3)
            x.metric("AUC", f"{m['AUC']:.3f}")
            y.metric("Gini", f"{m['Gini']:.3f}")
            z.metric("KS", f"{m['KS']:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ROC curve**")
        fig = go.Figure()
        for (name, r), colour in zip(a["roc"].items(), (BLUE, ORANGE)):
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], name=name, mode="lines", line=dict(color=colour, width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", mode="lines", line=dict(color=MUTED, dash="dot", width=1)))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        st.plotly_chart(style(fig), width="stretch")
    with c2:
        st.markdown("**Default rate by score band**")
        bands = pd.DataFrame(a["bands"])
        fig = go.Figure(go.Bar(x=bands["band"], y=bands["bad_rate"], marker_color=BLUE,
                               text=[f"{v:.1%}" for v in bands["bad_rate"]], textposition="outside",
                               hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", title="Default rate")
        st.plotly_chart(style(fig), width="stretch")
        st.caption("Each band holds about 20% of applicants. The default rate falls steadily as the score rises.")

    st.markdown("**Score distribution: repaid vs defaulted**")
    h = a["score_hist"]
    mids = [(h["edges"][i] + h["edges"][i + 1]) / 2 for i in range(len(h["good"]))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mids, y=h["good"], name="Repaid", marker_color=BLUE, opacity=0.85))
    fig.add_trace(go.Bar(x=mids, y=h["bad"], name="Defaulted", marker_color=ORANGE, opacity=0.9))
    fig.update_layout(barmode="overlay", xaxis_title="Score", yaxis_title="Applicants")
    st.plotly_chart(style(fig, 300), width="stretch")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Decisions on the hold-out**")
        dec = pd.DataFrame(a["decisions"]).set_index("decision").reindex(["Approve", "Refer", "Decline"])
        dec = dec.assign(share=dec["share"].map("{:.1%}".format), bad_rate=dec["bad_rate"].map("{:.1%}".format))
        st.dataframe(dec.rename(columns={"n": "Applicants", "bad_rate": "Default rate", "share": "Share"}),
                     width="stretch")
    with d2:
        st.markdown("**Scorecard scale**")
        s = a["scale"]
        st.markdown(f"- {s['base_score']:.0f} points = {s['base_odds_good_to_bad']:.0f}:1 good-to-bad odds\n"
                    f"- every {s['pdo']:.0f} points doubles the odds\n"
                    f"- band cut-offs: {', '.join(str(round(c)) for c in a['band_cutoffs'])}")

    st.markdown("**Scorecard: how much each factor can move the score**")
    pts = pd.DataFrame(get_report()["points_table"])
    spread = (pts.groupby("feature")["points"].agg(lambda s: s.max() - s.min())
              .sort_values().tail(12))
    fig = go.Figure(go.Bar(x=spread.values, y=[LABELS.get(f, f) for f in spread.index], orientation="h",
                           marker_color=BLUE, hovertemplate="%{y}: %{x:.0f} pts<extra></extra>"))
    fig.update_xaxes(title="Points between best and worst bin")
    st.plotly_chart(style(fig, 420), width="stretch")


# ── Page 3: monitoring ───────────────────────────────────────────────────────
def psi_kind(psi: float) -> str:
    return "Stable" if psi < 0.10 else "Monitor" if psi < 0.25 else "Alert"


def page_monitoring() -> None:
    a = get_assets()
    st.header("Monitoring")
    st.caption("Population Stability Index (PSI) compares live applicants with the training population. "
               "Below 0.10 is stable, 0.10 to 0.25 needs watching, above 0.25 triggers a model review.")
    st.info("Demo: the hold-out is altered to simulate a changing applicant population. In production, "
            "the same checks run monthly on real applications.")
    names = [d["scenario"] for d in a["drift"]]
    d = a["drift"][names.index(st.radio("Scenario", names, horizontal=True, key="scenario"))]

    kind = psi_kind(d["score_psi"]["PSI"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Score PSI", f"{d['score_psi']['PSI']:.3f}")
    c2.markdown(f"**Status**<br>{status_chip(kind, kind)}", unsafe_allow_html=True)
    c3.metric("Average score", d["mean_score"])
    action = {"Stable": "No action. Continue monthly monitoring.",
              "Monitor": "Investigate which inputs moved and check recent approval and arrears trends.",
              "Alert": "Model review required: confirm the shift is real, then recalibrate or retrain."}[kind]
    st.markdown(f"**Recommended action:** {action}")

    l, r = st.columns(2)
    with l:
        st.markdown("**PSI by input**")
        fp = pd.Series(d["feature_psi"]).sort_values()
        fig = go.Figure(go.Bar(x=fp.values, y=[LABELS.get(f, f) for f in fp.index], orientation="h",
                               marker_color=BLUE, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        for x in (0.10, 0.25):
            fig.add_vline(x=x, line=dict(color=MUTED, dash="dot", width=1))
        fig.update_xaxes(title="PSI (dotted lines: 0.10 and 0.25)")
        st.plotly_chart(style(fig), width="stretch")
    with r:
        st.markdown("**Applicants by score band**")
        mix = d["band_mix"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=BAND_ORDER, y=[a["train_band_mix"][b] for b in BAND_ORDER], name="Training", marker_color=BLUE))
        fig.add_trace(go.Bar(x=BAND_ORDER, y=[mix[b] for b in BAND_ORDER], name="Live", marker_color=ORANGE))
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%", title="Share of applicants")
        st.plotly_chart(style(fig), width="stretch")


# ── Page 4: roadmap ──────────────────────────────────────────────────────────
def page_roadmap() -> None:
    st.header("Roadmap and onboarding")
    st.markdown("The scoring engine, decision layer and monitoring are shared. Each lending product plugs in as a "
                "**profile**: its own inputs, policy rules, score bands and model.")
    cards = st.columns(4)
    for col, (name, state, kind, note) in zip(cards, [
        ("Micro-enterprise", "Working demo", "Approve", "Benchmark-trained; retrain on LOLC data"),
        ("SME business loans", "Next", "Monitor", "Adds financial ratios, collateral and cash-flow rules"),
        ("Gold loans", "Planned", "Monitor", "Collateral-led: weight, purity, loan-to-value"),
        ("Vehicle / leasing", "Planned", "Monitor", "Asset value, deposit, asset age"),
    ]):
        with col:
            st.markdown(f"**{name}**")
            st.markdown(status_chip(state, kind), unsafe_allow_html=True)
            st.caption(note)

    st.subheader("What changes when LOLC data arrives")
    spec = pd.DataFrame(FEATURE_SPEC, columns=["Feature", "Group", "Benchmark meaning", "LOLC source"])
    summary = (spec.groupby("Group").agg(Features=("Feature", "count"), Source=("LOLC source", lambda s: " / ".join(sorted(set(s)))))
               .reset_index())
    st.dataframe(summary, hide_index=True, width="stretch")

    st.subheader("Onboarding plan")
    st.markdown(
        "1. **Data request**: 24+ months of disbursements and monthly repayment status; application records; bureau pulls at application.\n"
        "2. **Definitions**: agree the default definition (for example 90+ days past due within 12 months) and policy thresholds.\n"
        "3. **Rebuild features** from LOLC fields; add loan-officer cash-flow inputs.\n"
        "4. **Retrain and validate** on an out-of-time sample; report AUC, Gini, KS and band default rates.\n"
        "5. **Shadow run** alongside current decisions, then champion/challenger live.\n"
        "6. **Monitor** monthly with PSI and band default rates.")

    st.subheader("Known limitations of this demo")
    st.markdown(
        "- Trained on Home Credit benchmark data, so figures describe the method and not LOLC's portfolio.\n"
        "- The benchmark has only approved applicants, so declined cases are not represented.\n"
        "- The three anonymised scores stand in for a bureau score; without them the same scorecard reaches an AUC of 0.69 "
        "instead of 0.75, so bureau data quality will drive results.\n"
        "- Policy thresholds are placeholders pending LOLC's credit policy.\n"
        "- Gender and family status are not used by the model.")


# ── Shell ─────────────────────────────────────────────────────────────────────
PAGES = {"Score an applicant": page_score, "Model performance": page_performance,
         "Monitoring": page_monitoring, "Roadmap and onboarding": page_roadmap}

with st.sidebar:
    st.title("Credit Scoring MVP")
    st.caption("Micro-lending · PKR")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Illustrative demo. Trained on a public benchmark dataset; to be retrained on LOLC data at onboarding.")

PAGES[page]()
