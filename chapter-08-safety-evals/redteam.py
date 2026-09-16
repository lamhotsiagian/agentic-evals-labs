"""Red-Team attack suite across 7 threat categories."""

from __future__ import annotations
from typing import Any, Dict, List
from shared.models.schemas import EvaluationCase


from shared.datasets.loader import load_redteam_attack_cases


def load_full_redteam_dataset() -> List[Dict[str, Any]]:
    """Loads benchmark adversarial test vectors directly from disk."""
    return load_redteam_attack_cases()
