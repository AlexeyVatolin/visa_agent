"""Output-side PII redaction tokens and redact_output_pii()."""

import re

from guardrails._pii import CARD_RE, EMAIL_RE, PHONE_RE

# Distinct tokens from the input-side labels so a leak is visually obvious.
_OUT_EMAIL_TOKEN = "[REDACTED-EMAIL]"
_OUT_PHONE_TOKEN = "[REDACTED-PHONE]"
_OUT_CARD_TOKEN = "[REDACTED-CARD]"

# Strip input-side tokens if they somehow leak back into the output.
_INPUT_TOKEN_RE = re.compile(r"\[(EMAIL|PHONE|CARD)\]")


def redact_output_pii(text: str) -> tuple[str, int, list[str]]:
    """Strip any PII tokens the model leaked. Returns (cleaned, count, concerns)."""
    concerns: list[str] = []
    cleaned = text
    n_total = 0

    cleaned, n = CARD_RE.subn(_OUT_CARD_TOKEN, cleaned)
    if n:
        concerns.append(f"credit-card numbers ({n})")
        n_total += n
    cleaned, n = EMAIL_RE.subn(_OUT_EMAIL_TOKEN, cleaned)
    if n:
        concerns.append(f"email addresses ({n})")
        n_total += n
    cleaned, n = PHONE_RE.subn(_OUT_PHONE_TOKEN, cleaned)
    if n:
        concerns.append(f"phone numbers ({n})")
        n_total += n
    cleaned, n = _INPUT_TOKEN_RE.subn("[REDACTED]", cleaned)
    if n:
        concerns.append(f"input-redaction tokens leaked ({n})")
        n_total += n

    return cleaned, n_total, concerns
