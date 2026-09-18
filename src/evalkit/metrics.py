from __future__ import annotations

import re


def exact_match(predicted: str, expected: str) -> float:
    return 1.0 if predicted.strip() == expected.strip() else 0.0


def contains(predicted: str, expected: str) -> float:
    return 1.0 if expected.strip().lower() in predicted.strip().lower() else 0.0


def regex_match(predicted: str, pattern: str) -> float:
    return 1.0 if re.search(pattern, predicted, re.I) else 0.0


def groundedness(predicted: str, context: list[str]) -> float:
    """Cheap lexical groundedness: fraction of predicted sentences that overlap context tokens."""
    if not predicted.strip():
        return 0.0
    ctx = set(" ".join(context).lower().split())
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", predicted) if s.strip()]
    if not sentences:
        return 0.0
    hits = 0
    for sentence in sentences:
        tokens = set(sentence.lower().split())
        if not tokens:
            continue
        overlap = len(tokens & ctx) / len(tokens)
        if overlap >= 0.3:
            hits += 1
    return hits / len(sentences)


def agreement(model_scores: list[float], human_scores: list[float], tolerance: float = 0.15) -> float:
    pairs = [(m, h) for m, h in zip(model_scores, human_scores) if h is not None]
    if not pairs:
        return 0.0
    return sum(abs(m - h) <= tolerance for m, h in pairs) / len(pairs)
