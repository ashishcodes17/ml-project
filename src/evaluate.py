from __future__ import annotations

from typing import Iterable


def precision_at_k(retrieved_ids: list[str], relevant_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    if not top_k:
        return 0.0
    correct = sum(1 for item in top_k if item in relevant)
    return correct / k
