"""Dataset loaders that parse and validate real physical data files from disk."""

from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List
from shared.models.schemas import EvaluationCase

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_customer_support_cases(count: int = 50) -> List[EvaluationCase]:
    """Loads real customer support evaluation cases from customer_support_50.jsonl."""
    file_path = os.path.join(DATA_DIR, "customer_support_50.jsonl")
    cases: List[EvaluationCase] = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Real dataset file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(EvaluationCase(**data))
            if len(cases) >= count:
                break

    return cases


def load_travel_planner_cases() -> List[EvaluationCase]:
    """Loads real travel planning constraint scenarios from travel_planner_tasks.json."""
    file_path = os.path.join(DATA_DIR, "travel_planner_tasks.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Real dataset file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    return [EvaluationCase(**item) for item in raw_list]


def load_ecommerce_tool_cases() -> List[EvaluationCase]:
    """Loads real e-commerce tool evaluation cases from ecommerce_db.json."""
    file_path = os.path.join(DATA_DIR, "ecommerce_db.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Real dataset file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        db_data = json.load(f)
    return [EvaluationCase(**item) for item in db_data.get("tool_evaluation_cases", [])]


def load_it_helpdesk_trajectories() -> List[Dict[str, Any]]:
    """Loads multi-turn IT helpdesk diagnosis trajectories from helpdesk_trajectories.jsonl."""
    file_path = os.path.join(DATA_DIR, "helpdesk_trajectories.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Real dataset file not found at: {file_path}")

    trajectories = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    return trajectories


def load_judge_benchmark_cases() -> List[Dict[str, Any]]:
    """Loads human-annotated golden benchmark cases from human_benchmark_judge.jsonl."""
    file_path = os.path.join(DATA_DIR, "human_benchmark_judge.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Real dataset file not found at: {file_path}")

    benchmarks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                benchmarks.append(json.loads(line))
    return benchmarks


def load_rag_enterprise_corpus() -> List[Dict[str, str]]:
    """Loads real enterprise markdown documents from enterprise_knowledge_base/."""
    kb_dir = os.path.join(DATA_DIR, "enterprise_knowledge_base")
    if not os.path.exists(kb_dir):
        raise FileNotFoundError(f"Enterprise knowledge base directory not found at: {kb_dir}")

    documents = []
    # Known doc files to load in order
    doc_files = [
        ("hr_policy.md", "DOC-HR-101", "HR"),
        ("security_policy.md", "DOC-SEC-202", "Security"),
        ("finance_refund_policy.md", "DOC-FIN-303", "Finance"),
        ("engineering_runbook.md", "DOC-ENG-404", "Engineering"),
    ]

    for fname, doc_id, dept in doc_files:
        fpath = os.path.join(kb_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                # Extract first markdown heading as title
                m = re.search(r"^#\s+(.*)", content, re.MULTILINE)
                title = m.group(1).strip() if m else fname.replace(".md", "").title()
                documents.append({
                    "doc_id": doc_id,
                    "department": dept,
                    "title": title,
                    "content": content,
                })

    return documents


def load_redteam_attack_cases() -> List[Dict[str, Any]]:
    """Loads adversarial attacks from redteam_adversarial_suite.jsonl."""
    file_path = os.path.join(DATA_DIR, "redteam_adversarial_suite.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Real dataset file not found at: {file_path}")

    attacks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                attacks.append(json.loads(line))
    return attacks


def load_chaos_scenarios() -> List[Dict[str, Any]]:
    """Loads chaos experiment workloads and fault injection configurations."""
    workload_file = os.path.join(DATA_DIR, "chaos_workload.jsonl")
    workloads = []
    if os.path.exists(workload_file):
        with open(workload_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    workloads.append(json.loads(line))

    return [
        {"fault": "tool_timeout", "description": "Injects 5000ms delay into primary database tool", "active": True},
        {"fault": "http_500", "description": "Returns HTTP 500 Internal Error on first attempt", "active": True},
        {"fault": "invalid_json", "description": "Returns malformed JSON syntax in tool response", "active": False},
        {"fault": "corrupted_context", "description": "Injects random character noise into prompt context", "active": False},
        {"workload_count": len(workloads)},
    ]
