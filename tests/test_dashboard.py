"""Smoke tests: every dashboard page renders and the scoring form responds."""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "dashboard.py"
ARTEFACTS = [ROOT / "models" / "micro_scorecard.pkl", ROOT / "models" / "micro_xgb_challenger.pkl",
             ROOT / "reports" / "demo_assets.json", ROOT / "reports" / "micro_scorecard_report.json"]

pytestmark = pytest.mark.skipif(not all(p.exists() for p in ARTEFACTS), reason="demo artefacts not built")

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def run_page(page: str):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    if page != "Score an applicant":
        at.sidebar.radio[0].set_value(page).run()
    return at


@pytest.mark.parametrize("page", ["Score an applicant", "Model performance", "Monitoring", "Roadmap and onboarding"])
def test_every_page_renders(page):
    at = run_page(page)
    assert not at.exception
    assert at.header, "page header missing"


def test_example_changes_decision():
    at = run_page("Score an applicant")
    at.selectbox[0].set_value("Severe overdue on bureau").run()
    assert not at.exception
    assert any("Decline" in m.value for m in at.markdown)

    at.selectbox[0].set_value("Repeat borrower, clean record").run()
    assert not at.exception
    assert any("Approve" in m.value for m in at.markdown)


def test_monitoring_scenarios():
    at = run_page("Monitoring")
    at.radio(key="scenario").set_value("Severe shift - income shock").run()
    assert not at.exception
    assert any("Alert" in m.value for m in at.markdown)
