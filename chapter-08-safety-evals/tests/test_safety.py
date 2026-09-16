"""Unit tests for Chapter 8 Safety & Red-Teaming."""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTER_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(CHAPTER_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CHAPTER_DIR)

for m in ("agent", "evaluator", "pipeline", "tools", "engine", "judge", "calibration", "system", "retriever", "chaos", "resilient_agent", "eval_platform", "reporter", "target_agent", "redteam"):
    sys.modules.pop(m, None)

from target_agent import BankingSupportAgent
from redteam import load_full_redteam_dataset
from evaluator import SafetyEvaluator


def test_protected_agent_resists_attacks():
    agent = BankingSupportAgent(guardrails_enabled=True)
    evaluator = SafetyEvaluator()
    attacks = load_full_redteam_dataset()

    res = evaluator.evaluate_suite(attacks, agent)
    assert res["total_tests"] == len(attacks)
    assert res["safety_score"] == 100.0
    assert res["failed"] == 0


def test_unprotected_agent_reveals_vulnerabilities():
    agent = BankingSupportAgent(guardrails_enabled=False)
    evaluator = SafetyEvaluator()
    attacks = load_full_redteam_dataset()

    res = evaluator.evaluate_suite(attacks, agent)
    assert res["failed"] > 0
    assert res["safety_score"] < 100.0
