"""PII detection and redaction patterns."""

import re

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

PHONE_RE = re.compile(r"(?<!\d)(?!\d{4}-\d{2}-\d{2}\b)\+?(?:\d[\s\-()]?){7,14}\d(?!\d)")

CARD_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")


def redact_pii(text: str) -> tuple[str, int]:
    """Replace PII tokens with labels. Returns (cleaned, count)."""
    count = 0
    # Order matters: cards before phones (cards are also long digit runs).
    for pattern, label in ((CARD_RE, "[CARD]"), (EMAIL_RE, "[EMAIL]"), (PHONE_RE, "[PHONE]")):
        text, n = pattern.subn(label, text)
        count += n
    return text, count
