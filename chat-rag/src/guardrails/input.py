"""Input guardrails: PII redaction and prompt-injection detection."""

from dataclasses import dataclass

from guardrails._injection import detect_injection
from guardrails._pii import redact_pii


@dataclass
class GuardrailResult:
    """Outcome of the input-guardrail pass.

    Fields:
        cleaned_text: input with PII tokens replaced ([EMAIL]/[PHONE]/[CARD]).
        pii_redactions: count of redactions made.
        injection_flagged: True if the input matches an injection heuristic.
        refusal_message: set only when the pipeline should short-circuit and
            reply with this text instead of invoking the graph further.
    """

    cleaned_text: str
    pii_redactions: int
    injection_flagged: bool
    refusal_message: str | None = None


def run_input_guardrails(text: str) -> GuardrailResult:
    """Apply all input guardrails to user text. See GuardrailResult for outputs."""
    cleaned, n = redact_pii(text)
    injection = detect_injection(cleaned)

    refusal: str | None = None
    if injection:
        refusal = (
            "Your message looks like an attempt to override my instructions. "
            "Try asking me about visas, travel documents, or embassy processes instead."
        )

    return GuardrailResult(
        cleaned_text=cleaned,
        pii_redactions=n,
        injection_flagged=injection,
        refusal_message=refusal,
    )
