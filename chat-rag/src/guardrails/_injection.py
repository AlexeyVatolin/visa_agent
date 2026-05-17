"""Prompt-injection detection heuristics."""

import re

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+|the\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(
        r"disregard\s+(?:your\s+|the\s+)?(?:prior\s+|previous\s+)?"
        r"(?:instructions|prompt|system)",
        re.IGNORECASE,
    ),
    re.compile(r"you\s+are\s+now\s+(?:a\s+|an\s+)?\w+", re.IGNORECASE),
    re.compile(r"system\s+prompt\s*:", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+)?(?:system\s+)?prompt", re.IGNORECASE),
]


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)
