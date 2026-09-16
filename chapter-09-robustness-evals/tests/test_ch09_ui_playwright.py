"""Playwright UI tests for Chapter 9 Streamlit App."""

import os
import pytest
from playwright.sync_api import sync_playwright, expect
from shared.testing_utils import run_streamlit_app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "app.py")


@pytest.mark.playwright
def test_chapter_09_robustness_ui():
    """Verify Chapter 9 Chaos Control Panel & Resilience Dashboard."""
    with run_streamlit_app(APP_PATH) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(base_url, timeout=45000)

            # 1. Header
            header = page.locator("h1")
            expect(header).to_contain_text("Chaos Testing & Reliability Evaluation", timeout=30000)

            # 2. Metrics
            expect(page.get_by_text("Normal Success").first).to_be_visible(timeout=60000)
            expect(page.get_by_text("Baseline under Chaos").first).to_be_visible()
            expect(page.get_by_text("Recovery Rate").first).to_be_visible()

            # 3. Columns
            expect(page.get_by_text("Baseline Agent (Fragile)").first).to_be_visible()
            expect(page.get_by_text("Resilient Agent (Hardened)").first).to_be_visible()

            # 4. Matrix
            expect(page.get_by_text("Reliability & Chaos Benchmark Summary").first).to_be_visible()

            browser.close()
