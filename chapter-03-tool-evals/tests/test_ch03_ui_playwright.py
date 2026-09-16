"""Playwright UI tests for Chapter 3 Streamlit App."""

import os
import pytest
from playwright.sync_api import sync_playwright, expect
from shared.testing_utils import run_streamlit_app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "app.py")


@pytest.mark.playwright
def test_chapter_03_tool_ui():
    """Verify Chapter 3 Tool Trace Viewer and Failure Panel."""
    with run_streamlit_app(APP_PATH) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(base_url, timeout=45000)

            # 1. Header
            header = page.locator("h1")
            expect(header).to_contain_text("Tool Calling Evaluation Harness", timeout=30000)

            # 2. Metrics
            expect(page.get_by_text("Tool Selection").first).to_be_visible(timeout=60000)
            expect(page.get_by_text("Argument Accuracy").first).to_be_visible()
            expect(page.get_by_text("Recovery Rate").first).to_be_visible()

            # 3. Tool Trace Viewer section
            expect(page.get_by_text("Tool Execution Trace Viewer").first).to_be_visible()
            expect(page.get_by_text("Agent Final Answer:").first).to_be_visible()

            # 4. Failure panel & Case Study table
            expect(page.get_by_text("Tool Failure & Recovery Panel").first).to_be_visible()
            expect(page.get_by_text("Fault Injection Matrix").first).to_be_visible()

            browser.close()
