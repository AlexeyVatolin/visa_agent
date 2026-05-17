"""Visa-domain scope keyword check."""

import re

_BASE_KEYWORDS: frozenset[str] = frozenset(
    {
        "visa",
        "schengen",
        "passport",
        "appointment",
        "embassy",
        "consulate",
        "application",
        "apply",
        "applying",
        "biometric",
        "biometrics",
        "residence",
        "permit",
        "tourist",
        "vfs",
        "extension",
        "renewal",
        "refusal",
        "refused",
        "processing",
        "interview",
        "sponsor",
        "sponsorship",
        "invitation",
        "immigration",
        "migrate",
        "migrant",
        "entry",
        "transit",
        "travel",
        "document",
        "documents",
        "fee",
        "fees",
        "insurance",
    }
)

def check_scope(text: str) -> bool:
    """Return True if the input is *out of scope* (no scope keyword found)."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return not (words & _BASE_KEYWORDS)
