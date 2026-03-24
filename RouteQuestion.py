import json
import re
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class RouteQuestion:
    route: str
    score: int
    normalized_question: str


def normalize_question(question: str) -> str:
    question = question.lower().strip()
    question = re.sub(r"[^a-z0-9\s]", " ", question)
    question = re.sub(r"\s+", " ", question)
    return question

def is_unique_party_count_question(question: str) -> bool:
    q = normalize_question(question)

    count_signals = [
        "how many",
        "number of",
        "count of",
    ]

    uniqueness_signals = [
        "unique",
        "distinct",
    ]

    entity_signals = [
        "party", "parties",
        "person", "persons",
        "company", "companies",
        "entity", "entities",
        "defendant", "defendants",
        "claimant", "claimants",
    ]

    has_count = any(s in q for s in count_signals)
    has_uniqueness = any(s in q for s in uniqueness_signals)
    has_entity = any(s in q for s in entity_signals)

    return has_count and has_uniqueness and has_entity


def route_question(question: str) -> RouteQuestion:
    q = normalize_question(question)
    score = 0

    if "main party" in q or "main parties" in q:
        score += 3

    if "party" in q:
        score += 1

    overlap_terms0 = ["person", "company", "party", "low", "entity", "individual"]
    if any(term in q for term in overlap_terms0):
        score -= 3

    overlap_terms = ["common", "shared", "same"]
    if any(term in q for term in overlap_terms):
        score += 2

    both_terms = ["both", "two cases", "two proceedings", "two matters"]
    if any(term in q for term in both_terms):
        score += 2

    phrase_patterns = [
        "common to both",
        "common in both",
        "appear in both",
        "present in both",
    ]
    if any(p in q for p in phrase_patterns):
        score += 3

    route = "main_party_overlap" if score >= 5 and "party" in q else "default_rag"

    return RouteQuestion(
        route=route,
        score=score,
        normalized_question=q,
    )


