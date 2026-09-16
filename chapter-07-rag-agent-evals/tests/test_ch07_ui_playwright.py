"""Playwright UI tests for Chapter 7 Streamlit App."""

import os
import pytest
from playwright.sync_api import sync_playwright, expect
from shared.testing_utils import run_streamlit_app

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "app.py")


@pytest.mark.playwright
def test_chapter_07_rag_ui():
    """Verify Chapter 7 RAG Explorer UI."""
    with run_streamlit_app(APP_PATH) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto(base_url, timeout=45000)

            # 1. Header
            header = page.locator("h1").first
            expect(header).to_contain_text("RAG Evaluation Explorer", timeout=30000)

            # 2. Metrics
            expect(page.get_by_text("Retrieval Precision").first).to_be_visible(timeout=60000)
            expect(page.get_by_text("Faithfulness").first).to_be_visible()
            expect(page.get_by_text("Groundedness").first).to_be_visible()

            # 3. Question & Answer
            expect(page.get_by_text("Question & Augmented Answer").first).to_be_visible()
            expect(page.get_by_text("Generated Answer:").first).to_be_visible()

            # 4. Retrieved Documents
            expect(page.get_by_text("Retrieved Documents").first).to_be_visible()

            browser.close()
