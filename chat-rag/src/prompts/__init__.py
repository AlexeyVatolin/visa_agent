from prompts.answer import ANSWER_PROMPT
from prompts.classify import CLASSIFY_PROMPT
from prompts.detect import COUNTRY_DETECT_PROMPT, KNOWN_COUNTRIES, slug_to_collection

__all__ = [
    "ANSWER_PROMPT",
    "CLASSIFY_PROMPT",
    "COUNTRY_DETECT_PROMPT",
    "KNOWN_COUNTRIES",
    "slug_to_collection",
]
