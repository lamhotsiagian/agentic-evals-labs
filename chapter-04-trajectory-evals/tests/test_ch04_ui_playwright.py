"""Playwright UI tests for Chapter 4 Streamlit App."""

import os
import pytest
from playwright.sync_api import sync_playwright, expect
from shared.testing_utils import run_streamlit_app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "app.py")


@pytest.mark.playwright
def test_chapter_04_trajectory_ui():
    """Verify Chapter 4 Timeline UI and Trajectory Score."""
    with run_streamlit_app(APP_PATH) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(base_url, timeout=45000)

            # 1. Header
            header = page.locator("h1")
            expect(header).to_contain_text("Agent Trajectory Evaluation Engine", timeout=30000)

            # 2. Metrics
            expect(page.get_by_text("Trajectory Score").first).to_be_visible(timeout=60000)
            expect(page.get_by_text("Step Success").first).to_be_visible()
            expect(page.get_by_text("Recovery Rate").first).to_be_visible()

            # 3. Timeline & Steps
            expect(page.get_by_text("Step-by-Step Trajectory Timeline").first).to_be_visible()
            expect(page.get_by_text("parse_user_ticket").first).to_be_visible()
            expect(page.get_by_text("Final Resolution:").first).to_be_visible()

            # 4. Diagnostic summary
            expect(page.get_by_text("Diagnostic Summary").first).to_be_visible()

            browser.close()
