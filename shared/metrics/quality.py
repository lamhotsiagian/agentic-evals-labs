"""Quality metrics for agent evaluations."""

from __future__ import annotations
import re
from typing import List, Set


STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "our",
    "for", "to", "in", "on", "at", "by", "with", "about", "what", "should", "do", "how"
}


def _tokenize(text: str, remove_stopwords: bool = False) -> Set[str]:
    tokens = set(re.findall(r"\w+", text.lower()))
    if remove_stopwords:
        tokens = {t for t in tokens if t not in STOPWORDS and len(t) > 2}
    return tokens


def _stem(word: str) -> str:
    """Simple suffix stripper for basic stemming."""
    for suffix in ["ing", "ed", "tion", "tions", "es", "s"]:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def compute_exact_match(predicted: str, reference: str) -> float:
    """Returns 1.0 if normalized strings match exactly, else 0.0."""
    return 1.0 if predicted.strip().lower() == reference.strip().lower() else 0.0


def compute_substring_match(predicted: str, keywords: List[str]) -> float:
    """Returns fraction of target keywords present in predicted text."""
    if not keywords:
        return 1.0
    text = predicted.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return float(matches) / len(keywords)


def compute_similarity(predicted: str, reference: str) -> float:
    """Jaccard word-overlap similarity between prediction and reference (0.0 to 1.0)."""
    tokens_pred = _tokenize(predicted)
    tokens_ref = _tokenize(reference)
    if not tokens_pred and not tokens_ref:
        return 1.0
    if not tokens_pred or not tokens_ref:
        return 0.0
    intersection = tokens_pred.intersection(tokens_ref)
    union = tokens_pred.union(tokens_ref)
    return len(intersection) / len(union)


def detect_hallucination(predicted: str, context: str) -> bool:
    """
    Checks if predicted text asserts specific claims/numbers absent in context.
    Returns True if hallucination likely detected.
    """
    pred_numbers = set(re.findall(r"\b\d+\b", predicted))
    context_numbers = set(re.findall(r"\b\d+\b", context))
    unsupported_numbers = pred_numbers - context_numbers
    return len(unsupported_numbers) > 1


def compute_relevance(predicted: str, prompt: str) -> float:
    """Measures topical overlap between prompt requirements and prediction."""
    prompt_tokens = {_stem(t) for t in _tokenize(prompt, remove_stopwords=True)}
    if not prompt_tokens:
        return 1.0
    pred_tokens = {_stem(t) for t in _tokenize(predicted, remove_stopwords=True)}
    overlap = prompt_tokens.intersection(pred_tokens)
    
    # Also reward domain resolution words
    domain_terms = {"refund", "support", "billing", "assist", "account", "charge", "payment"}
    domain_overlap = domain_terms.intersection(pred_tokens)
    
    score = (len(overlap) / max(1, len(prompt_tokens))) * 0.7 + (min(2, len(domain_overlap)) / 2.0) * 0.3
    return min(1.0, round(score, 2))


def compute_faithfulness(answer: str, context: str) -> float:
    """
    RAGAS-style faithfulness metric:
    Ratio of claims in the answer supported by the retrieved context.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0
    
    context_lower = context.lower()
    supported = 0
    for sentence in sentences:
        words = [w for w in re.findall(r"\w+", sentence.lower()) if len(w) > 3]
        if not words:
            supported += 1
            continue
        word_matches = sum(1 for w in words if w in context_lower)
        if (word_matches / len(words)) >= 0.5:
            supported += 1

    return supported / len(sentences)
