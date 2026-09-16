"""Playwright UI tests for Chapter 10 Master Platform."""

import os
import pytest
from playwright.sync_api import sync_playwright, expect
from shared.testing_utils import run_streamlit_app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "app.py")


@pytest.mark.playwright
def test_chapter_10_platform_ui():
    """Verify Chapter 10 Production Platform Master Dashboard and Tabs."""
    with run_streamlit_app(APP_PATH) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(base_url, timeout=45000)

            # 1. Header
            header = page.locator("h1").first
            expect(header).to_contain_text("Agent Evaluation Platform", timeout=30000)

            # 2. Executive Scorecard Metrics
            expect(page.get_by_text("Executive Scorecard").first).to_be_visible(timeout=60000)
            expect(page.get_by_text("Task Success").first).to_be_visible()
            expect(page.get_by_text("Safety Score").first).to_be_visible()
            expect(page.get_by_text("CI/CD Gate").first).to_be_visible()

            # 3. Tab navigation
            page.get_by_role("tab", name="Trajectories").click()
            expect(page.get_by_text("Enterprise Support Agent Trajectory Flow").first).to_be_visible()

            page.get_by_role("tab", name="Failures").click()
            expect(page.get_by_text("Failure Classification & Root-Cause Matrix").first).to_be_visible()

            page.get_by_role("tab", name="Regression & CI/CD").click()
            expect(page.get_by_text("CI/CD Quality Gate Status: PASSED").first).to_be_visible()

            browser.close()
