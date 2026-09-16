"""Playwright UI tests for Chapter 2 Streamlit App."""

import os
import pytest
from playwright.sync_api import sync_playwright, expect
from shared.testing_utils import run_streamlit_app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "app.py")


@pytest.mark.playwright
def test_chapter_02_architecture_ui():
    """Verify Chapter 2 Streamlit UI renders visual pipeline and node inspector."""
    with run_streamlit_app(APP_PATH) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(base_url, timeout=45000)

            # 1. Verify Page Title
            header = page.locator("h1")
            expect(header).to_contain_text("Agent Architecture Evaluation", timeout=30000)

            # 2. Verify Component Metrics
            expect(page.get_by_text("Planning Accuracy").first).to_be_visible(timeout=60000)
            expect(page.get_by_text("Execution Accuracy").first).to_be_visible()
            expect(page.get_by_text("E2E Success").first).to_be_visible()

            # 3. Verify Visual Pipeline Nodes
            expect(page.get_by_text("[PLANNER]").first).to_be_visible()
            expect(page.get_by_text("[EXECUTOR]").first).to_be_visible()
            expect(page.get_by_text("[VERIFIER]").first).to_be_visible()

            # 4. Click Inspector Tabs
            page.get_by_role("tab", name="Planner Details").click()
            expect(page.get_by_text("Attempt 0 — ✓ Success").first).to_be_visible()

            page.get_by_role("tab", name="Verifier Details").click()
            expect(page.get_by_text("Attempt 0 — ✓ Verified").first).to_be_visible()

            browser.close()
