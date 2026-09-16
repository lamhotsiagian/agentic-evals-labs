"""Playwright UI tests for Chapter 1 Streamlit Dashboard."""

import os
import pytest
from playwright.sync_api import sync_playwright, expect
from shared.testing_utils import run_streamlit_app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "app.py")


@pytest.mark.playwright
def test_chapter_01_dashboard_ui():
    """Verify Chapter 1 Streamlit Dashboard loads and renders evaluation results."""
    with run_streamlit_app(APP_PATH) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(base_url, timeout=45000)

            # 1. Verify Page Title and Header
            header = page.locator("h1")
            expect(header).to_contain_text("Agent Evaluation Dashboard", timeout=30000)

            # 2. Verify KPI Metric Cards exist
            expect(page.get_by_text("Tasks Evaluated").first).to_be_visible(timeout=60000)
            expect(page.get_by_text("Success Rate").first).to_be_visible()

            # 3. Verify Tabs are present
            expect(page.get_by_role("tab", name="View Test Cases")).to_be_visible()
            expect(page.get_by_role("tab", name="Model Comparison")).to_be_visible()

            # 4. Click Model Comparison tab and verify Case Study table renders
            page.get_by_role("tab", name="Model Comparison").click()
            expect(page.get_by_text("Model Quality & Latency Tradeoff").first).to_be_visible()
            expect(page.get_by_text("Qwen2.5:3B").first).to_be_visible()

            # 5. Click View Test Cases tab
            page.get_by_role("tab", name="View Test Cases").click()
            expect(page.locator("div[data-testid='stDataFrame']").first).to_be_visible()

            browser.close()
