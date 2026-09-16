"""Playwright UI tests for Chapter 5 Streamlit App."""

import os
import pytest
from playwright.sync_api import sync_playwright, expect
from shared.testing_utils import run_streamlit_app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "app.py")


@pytest.mark.playwright
def test_chapter_05_judge_ui():
    """Verify Chapter 5 Judge Scorecard and Meta-Evaluation calibration view."""
    with run_streamlit_app(APP_PATH) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(base_url, timeout=45000)

            # 1. Header
            header = page.locator("h1")
            expect(header).to_contain_text("LLM-as-a-Judge Evaluation", timeout=30000)

            # 2. Scorecard metrics & tables
            expect(page.get_by_text("Judge Scorecard").first).to_be_visible(timeout=60000)
            expect(page.get_by_text("Overall Score").first).to_be_visible()
            expect(page.get_by_text("View Judge Reasoning & Evidence Citations").first).to_be_visible()

            # 3. Meta-Evaluation calibration tab
            page.get_by_role("tab", name="Meta-Evaluation (Judge Calibration)").click()
            expect(page.get_by_text("Human Agreement").first).to_be_visible(timeout=60000)
            expect(page.get_by_text("Pearson Correlation").first).to_be_visible()
            expect(page.get_by_text("Benchmark Comparison Matrix").first).to_be_visible()

            browser.close()
