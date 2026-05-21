KNOWN_COUNTRIES = [
    "albania",
    "austria",
    "belgium",
    "bulgaria",
    "china",
    "croatia",
    "cyprus",
    "france",
    "germany",
    "greece",
    "hungary",
    "italy",
    "japan",
    "macedonia",
    "netherlands",
    "poland",
    "portugal",
    "romania",
    "slovakia",
    "slovenia",
    "spain",
    "sweden",
    "switzerland",
    "united_kingdom",
    "usa",
]

# Only slugs that differ from slug.title() / need special mapping.
_COLLECTION_OVERRIDES: dict[str, str] = {
    "bulgaria": "Bulgaria_Romania_Cyprus",
    "poland": "Poland_visa_D",
    "united_kingdom": "United_Kingdom",
    "usa": "USA",
}


def slug_to_collection(slug: str) -> str:
    return _COLLECTION_OVERRIDES.get(slug, slug.replace("_", " ").title().replace(" ", "_"))


COUNTRY_DETECT_PROMPT = (
    "You are a country extractor. Given a visa-related question in any language, "
    "identify which destination country the question is about.\n"
    "Reply with exactly one slug from this list:\n"
    f"{', '.join(KNOWN_COUNTRIES)}\n"
    "If the question does not mention a specific country, reply with: none\n"
    "Reply with the slug only — no explanation, no punctuation."
)
