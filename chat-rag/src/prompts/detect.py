KNOWN_COUNTRIES = [
    "albania",
    "austria",
    "belgium",
    "bulgaria",
    "croatia",
    "cyprus",
    "france",
    "germany",
    "greece",
    "hungary",
    "italy",
    "netherlands",
    "poland",
    "portugal",
    "romania",
    "slovenia",
    "spain",
    "sweden",
    "switzerland",
    "united_kingdom",
    "usa",
]

COUNTRY_DETECT_PROMPT = (
    "You are a country extractor. Given a visa-related question in any language, "
    "identify which destination country the question is about.\n"
    "Reply with exactly one slug from this list:\n"
    f"{', '.join(KNOWN_COUNTRIES)}\n"
    "If the question does not mention a specific country, reply with: none\n"
    "Reply with the slug only — no explanation, no punctuation."
)
