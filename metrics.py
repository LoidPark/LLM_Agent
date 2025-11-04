# metrics.py
from typing import List, Set, Tuple


def context_precision(labels: List[bool]) -> float:
    if not labels:
        return 0.0
    return sum(labels) / len(labels)


def context_recall(covered_facts: Set[int], total_facts: int) -> float:
    if total_facts == 0:
        return 0.0
    return len(covered_facts) / total_facts
