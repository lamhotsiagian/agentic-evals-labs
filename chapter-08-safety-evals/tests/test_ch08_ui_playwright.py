"""Playwright UI tests for Chapter 8 Streamlit App."""

import os
import pytest
from playwright.sync_api import sync_playwright, expect
from shared.testing_utils import run_streamlit_app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "app.py")


@pytest.mark.playwright
def test_chapter_08_safety_ui():
    """Verify Chapter 8 Red-Team Security Dashboard & Replay interface."""
    with run_streamlit_app(APP_PATH) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(base_url, timeout=45000)

            # 1. Header
            header = page.locator("h1")
            expect(header).to_contain_text("Agent Red-Team Security Evaluation", timeout=30000)

            # 2. Metrics
            expect(page.get_by_text("Total Tests").first).to_be_visible(timeout=60000)
            expect(page.get_by_text("Safety Score").first).to_be_visible()

            # 3. Breakdown & Replay
            expect(page.get_by_text("Vulnerability Category Breakdown").first).to_be_visible()
            expect(page.get_by_text("Red-Team Attack Replay").first).to_be_visible()
            expect(page.get_by_text("Verdict: PASSED").first).to_be_visible()

            browser.close()
