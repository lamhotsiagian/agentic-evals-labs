"""Headless UI smoke tests for the consolidated Streamlit app.

Replaces the ten per-chapter Playwright tests, which drove the old
per-chapter app.py dashboards that no longer exist. These use Streamlit's
own AppTest runner: no browser, no server, and no model (MOCK_LLM=1), so
they check that Home.py and every chapter page render and that switching
between pages in both directions never leaks one chapter's local modules
(agent.py, evaluator.py, ...) into another.
"""

import os
import glob

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = sorted(
    glob.glob(os.path.join(ROOT, "pages", "*.py")),
    key=lambda p: int(os.path.basename(p).split("_")[0]),
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")


def _run(path):
    at = AppTest.from_file(path, default_timeout=120)
    at.run()
    return at


def test_there_are_ten_chapter_pages():
    assert len(PAGES) == 10


def test_home_renders():
    at = _run(os.path.join(ROOT, "Home.py"))
    assert not at.exception


@pytest.mark.parametrize("page", PAGES, ids=lambda p: os.path.basename(p))
def test_each_page_renders(page):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_switching_pages_forward_and_backward_never_crashes():
    for page in PAGES + PAGES[::-1]:
        at = _run(page)
        assert not at.exception, (os.path.basename(page), [e.value for e in at.exception])


def test_ch1_live_chat_turn_is_scored():
    at = _run(os.path.join(ROOT, "pages", "1_Ch1_Evaluation_Foundations.py"))
    at.chat_input(key="ch1_chat_input").set_value("I was charged twice for my subscription").run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Correctness" in labels and "Hallucination" in labels


def test_ch9_experiment_runs_without_a_model():
    at = _run(os.path.join(ROOT, "pages", "9_Ch9_Robustness_Evals.py"))
    at.button(key="ch9_run").click().run()
    assert not at.exception
    assert any("Circuit breaker" in i.value for i in at.info)
