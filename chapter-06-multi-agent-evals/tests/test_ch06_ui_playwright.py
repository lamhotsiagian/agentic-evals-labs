"""Playwright UI tests for Chapter 6 Streamlit App."""

import os
import pytest
from playwright.sync_api import sync_playwright, expect
from shared.testing_utils import run_streamlit_app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "app.py")


@pytest.mark.playwright
def test_chapter_06_multi_agent_ui():
    """Verify Chapter 6 Multi-Agent Network visualization and message stream."""
    with run_streamlit_app(APP_PATH) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(base_url, timeout=45000)

            # 1. Header
            header = page.locator("h1")
            expect(header).to_contain_text("Multi-Agent Research System Evaluation", timeout=30000)

            # 2. Metrics
            expect(page.get_by_text("Messages").first).to_be_visible(timeout=60000)
            expect(page.get_by_text("Successful Handoffs").first).to_be_visible()
            expect(page.get_by_text("Handoff Success Rate").first).to_be_visible()

            # 3. Network Cards
            expect(page.get_by_text("Supervisor Agent").first).to_be_visible()
            expect(page.get_by_text("Researcher Agent").first).to_be_visible()
            expect(page.get_by_text("Analyst Agent").first).to_be_visible()
            expect(page.get_by_text("Synthesizer Agent").first).to_be_visible()

            # 4. Message Stream
            expect(page.get_by_text("Message Handoff Stream").first).to_be_visible()
            expect(page.get_by_text("Synthesizer Output").first).to_be_visible()

            browser.close()
