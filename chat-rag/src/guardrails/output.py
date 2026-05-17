"""Output guardrails: deterministic PII redaction and internal error scrubbing."""

import logging
from dataclasses import dataclass, field

from guardrails._internal_errors import rewrite_internal_error_leaks
from guardrails._output_tokens import redact_output_pii

logger = logging.getLogger(__name__)


@dataclass
class OutputGuardrailResult:
    """Outcome of the output-guardrail pass.

    ``response_text`` is always safe to render — PII and internal errors have
    been stripped. ``pii_concerns`` / ``internal_error_concerns`` are non-empty
    when something was redacted.
    """

    response_text: str
    pii_leaked: int = 0
    pii_concerns: list[str] = field(default_factory=list)
    internal_error_rewritten: bool = False
    internal_error_concerns: list[str] = field(default_factory=list)

    @property
    def has_concerns(self) -> bool:
        return bool(self.pii_concerns or self.internal_error_concerns)


def run_output_guardrails(response_text: str) -> OutputGuardrailResult:
    """Apply deterministic output guardrails and return a safe-to-render result."""
    cleaned, internal_rewritten, internal_concerns = rewrite_internal_error_leaks(response_text)
    cleaned, n_pii, pii_concerns = redact_output_pii(cleaned)

    if internal_rewritten or n_pii:
        logger.warning(
            "output_guardrail internal_rewritten=%s pii_leaked=%d",
            internal_rewritten,
            n_pii,
        )
    else:
        logger.info("output_guardrail clean")

    return OutputGuardrailResult(
        response_text=cleaned,
        pii_leaked=n_pii,
        pii_concerns=pii_concerns,
        internal_error_rewritten=internal_rewritten,
        internal_error_concerns=internal_concerns,
    )
