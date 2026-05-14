import json
from datetime import date, datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "extracted"

COUNTRY_FLAGS: dict[str, str] = {
    "Австрия": "🇦🇹",
    "Албания": "🇦🇱",
    "Бельгия": "🇧🇪",
    "Болгария,_Румыния,_Кипр": "🇧🇬🇷🇴🇨🇾",
    "Великобритания": "🇬🇧",
    "Венгрия": "🇭🇺",
    "Германия": "🇩🇪",
    "Греция": "🇬🇷",
    "Испания": "🇪🇸",
    "Италия": "🇮🇹",
    "Кипр": "🇨🇾",
    "Китай": "🇨🇳",
    "Нидерланды": "🇳🇱",
    "Польша_(visa_D_only)": "🇵🇱",
    "Португалия": "🇵🇹",
    "Румыния": "🇷🇴",
    "США": "🇺🇸",
    "Словения": "🇸🇮",
    "Франция": "🇫🇷",
    "Хорватия": "🇭🇷",
    "Швейцария": "🇨🇭",
    "Швеция": "🇸🇪",
    "Япония": "🇯🇵",
}


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_tourist_ids_by_country() -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for path in sorted((DATA_DIR / "02_visa_type").glob("*.jsonl")):
        records = load_jsonl(path)
        result[path.stem] = {
            r["message_id"]
            for r in records
            if r["extracted"]["visa_type"] == "tourist"
        }
    return result


def load_validity_by_country() -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    validity_dir = DATA_DIR / "04_visa_validity"
    for path in sorted(validity_dir.glob("*.jsonl")):
        country = path.stem
        records = load_jsonl(path)
        validity: dict[int, int] = {}
        for r in records:
            days = r["extracted"]["validity_days"]
            if days is not None:
                validity[r["message_id"]] = days
        result[country] = validity
    return result


def load_entries_by_country() -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {}
    for path in sorted((DATA_DIR / "03_entries").glob("*.jsonl")):
        records = load_jsonl(path)
        result[path.stem] = {
            r["message_id"]: r["extracted"]["entry_type"] for r in records
        }
    return result


def compute_tourist_stats() -> list[list]:
    cutoff = datetime.now() - timedelta(days=365)
    dates_dir = DATA_DIR / "01_dates"
    tourist_ids_by_country = load_tourist_ids_by_country()
    validity_by_country = load_validity_by_country()
    entries_by_country = load_entries_by_country()

    rows: list[list] = []
    for path in sorted(dates_dir.glob("*.jsonl")):
        country = path.stem
        records = load_jsonl(path)
        records = [
            r for r in records
            if datetime.fromisoformat(r["message_date"]) >= cutoff
        ]

        tourist_ids = tourist_ids_by_country.get(country)
        if tourist_ids is not None:
            records = [r for r in records if r["message_id"] in tourist_ids]

        total = len(records)
        if total == 0:
            continue

        wait_times: list[int] = []
        for r in records:
            ext = r["extracted"]
            if ext["submission_date"] and ext["ready_date"]:
                sub = date.fromisoformat(ext["submission_date"])
                ready = date.fromisoformat(ext["ready_date"])
                wait_times.append((ready - sub).days)

        approved = sum(
            1 for r in records if r["tag"] in ("#одобрено", "#одобрение")
        )
        approval_rate = approved / total * 100

        validity = validity_by_country.get(country, {})
        durations = [
            validity[r["message_id"]]
            for r in records
            if r["message_id"] in validity
        ]

        entries = entries_by_country.get(country, {})
        multi_count = sum(
            1
            for r in records
            if entries.get(r["message_id"]) == "multi"
        )

        flag = COUNTRY_FLAGS.get(country, "")
        rows.append([
            f"{flag} {country}".strip(),
            round(sum(wait_times) / len(wait_times)) if wait_times else None,
            total,
            round(approval_rate),
            round(sum(durations) / len(durations)) if durations else None,
            multi_count if multi_count > 0 else None,
        ])

    rows.sort(key=lambda r: r[2], reverse=True)
    return rows
