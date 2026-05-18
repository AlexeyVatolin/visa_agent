"""Internal-error scrubber: strips env vars, auth errors, provider names."""

import re

_INTERNAL_ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bMISTRAL_API_KEY\b"), "env-var leak"),
    (re.compile(r"\bANTHROPIC_API_KEY\b"), "env-var leak"),
    (re.compile(r"\bOPENAI_API_KEY\b"), "env-var leak"),
    (re.compile(r"\bHTTP Error \d+\b", re.IGNORECASE), "provider http error leak"),
    (re.compile(r"\bUnauthorized\b", re.IGNORECASE), "provider auth leak"),
    (re.compile(r"\bauthorization error\b", re.IGNORECASE), "provider auth leak"),
    (re.compile(r"\bprovider failure\b", re.IGNORECASE), "provider failure leak"),
    (re.compile(r"\bAPI key\b", re.IGNORECASE), "api-key leak"),
    (re.compile(r"\bchromadb\b", re.IGNORECASE), "internal db leak"),
    (re.compile(r"\bvectorstore\b", re.IGNORECASE), "internal infra leak"),
]


def rewrite_internal_error_leaks(text: str) -> tuple[str, bool, list[str]]:
    """Remove sentences that expose internal infrastructure details."""
    concerns: list[str] = []
    hit = False

    segments = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    kept: list[str] = []
    for segment in segments:
        stripped = segment.strip()
        if not stripped:
            continue
        segment_has_internal = False
        for pattern, label in _INTERNAL_ERROR_PATTERNS:
            if pattern.search(stripped):
                concerns.append(label)
                segment_has_internal = True
                hit = True
        if not segment_has_internal:
            kept.append(stripped)

    if not hit:
        return text, False, []

    cleaned = "\n\n".join(kept).strip()
    if not cleaned:
        cleaned = (
            "I couldn't retrieve all information right now, "
            "but I can still help with other available data."
        )
    return cleaned, True, sorted(set(concerns))
