"""Input guardrails: PII redaction, prompt-injection detection, visa scope check."""

from dataclasses import dataclass

from guardrails._injection import detect_injection
from guardrails._pii import redact_pii
from guardrails._scope import check_scope

@dataclass
class GuardrailResult:
    """Outcome of the input-guardrail pass.

    Fields:
        cleaned_text: input with PII tokens replaced ([EMAIL]/[PHONE]/[CARD]).
        pii_redactions: count of redactions made.
        injection_flagged: True if the input matches an injection heuristic.
        out_of_scope: True if no scope keyword was found.
        refusal_message: set only when the pipeline should short-circuit and
            reply with this text instead of invoking the graph further.
    """

    cleaned_text: str
    pii_redactions: int
    injection_flagged: bool
    out_of_scope: bool
    refusal_message: str | None = None


def run_input_guardrails(text: str) -> GuardrailResult:
    """Apply all input guardrails to user text. See GuardrailResult for outputs."""
    cleaned, n = redact_pii(text)
    injection = detect_injection(cleaned)
    out_of_scope = check_scope(cleaned)

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
        out_of_scope=out_of_scope,
        refusal_message=refusal,
    )
