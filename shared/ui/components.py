"""Shared Streamlit UI building blocks for the consolidated lab app.

These are the only pieces every chapter page has in common: a provider
status badge (so a down Ollama server is always visible, never silently
swapped for canned text), chat history rendering, and a metrics row.
Everything chapter-specific (what the agent does, how it is scored)
stays in that chapter's own page -- this module never fabricates output.
"""

from __future__ import annotations
import sys
from typing import Dict, Iterable, List, Optional

import streamlit as st

from shared.models.provider import (
    LLMProvider,
    DeterministicMockProvider,
    OllamaClient,
    provider_identity,
)


def provider_status_badge(provider: LLMProvider, location="sidebar") -> bool:
    """Render a badge showing the real backend in use. Returns True if live.

    Mock mode is shown as an explicit amber badge, never disguised as a
    live model. A real Ollama backend that fails is_available() is shown
    as red/unreachable -- it is never swapped out from under the caller.
    """
    target = st.sidebar if location == "sidebar" else st
    identity = provider_identity(provider)
    if isinstance(provider, DeterministicMockProvider):
        target.warning(f"MOCK MODE\n\n`{identity}` -- deterministic canned text, "
                        "for tests/offline dev only. Unset MOCK_LLM for live chat.")
        return False
    if isinstance(provider, OllamaClient):
        if provider.is_available():
            target.success(f"LIVE MODEL\n\n`{identity}`")
            return True
        target.error(f"UNREACHABLE\n\n`{identity}`\n\nStart it: `ollama serve`, "
                      "then pull the models this chapter needs (see README).")
        return False
    target.info(identity)
    return False


def render_message_history(history: Iterable[dict]) -> None:
    """Replay a list of {'role': 'user'|'assistant', 'content': str} turns."""
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            extra = turn.get("extra")
            if extra:
                extra()


def metrics_row(scores: Dict[str, float], passed: Optional[bool] = None, precision: int = 2) -> None:
    """Render a row of st.metric tiles from a {name: score} dict."""
    if not scores:
        return
    cols = st.columns(len(scores) + (1 if passed is not None else 0))
    for col, (name, value) in zip(cols, scores.items()):
        label = name.replace("_", " ").title()
        try:
            col.metric(label, f"{float(value):.{precision}f}")
        except (TypeError, ValueError):
            col.metric(label, str(value))
    if passed is not None:
        cols[-1].metric("Gate", "PASS" if passed else "FAIL")


def chapter_page_header(chapter_no: int, title: str, caption: str, icon: Optional[str] = None) -> None:
    st.set_page_config(page_title=f"Ch{chapter_no}: {title}", layout="wide")
    st.title(f"Chapter {chapter_no}: {title}")
    st.caption(caption)

# Every chapter directory reuses generic module names (evaluator.py,
# agent.py, tools.py, graders.py, ...). Streamlit re-executes a page's
# script on every navigation, but Python's sys.modules cache is process-
# global and persists across those re-runs -- so without this, opening
# Chapter 2 after Chapter 1 could silently import Chapter 1's cached
# "evaluator" module instead of Chapter 2's. Call this at the top of
# every chapter page before importing chapter-local modules.
CHAPTER_LOCAL_MODULE_NAMES = [
    "agent", "evaluator", "pipeline", "tools", "engine", "judge", "calibration",
    "system", "graders", "retriever", "chaos", "resilient_agent", "eval_platform", "reporter",
    "gateway", "target_agent", "redteam", "gate", "suites",
]


def load_chapter_modules(chapter_dir: str, extra_names: Optional[List[str]] = None) -> None:
    """Evict any previously-imported chapter-local module from sys.modules
    and make sure this chapter's own directory is first on sys.path, so
    the next `import evaluator` (etc.) resolves to THIS chapter's file."""
    names = list(CHAPTER_LOCAL_MODULE_NAMES)
    if extra_names:
        names += [n for n in extra_names if n not in names]
    for name in names:
        sys.modules.pop(name, None)
    if chapter_dir in sys.path:
        sys.path.remove(chapter_dir)
    sys.path.insert(0, chapter_dir)
