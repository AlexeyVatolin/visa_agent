import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import typer
from aiolimiter import AsyncLimiter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.output import NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from schemas import (
    AdditionalDocsResult,
    CountriesResult,
    DatesResult,
    EmploymentResult,
    EntriesResult,
    GroupResult,
    PriorSchengenResult,
    RejectionResult,
    RequestedVsGrantedResult,
    SponsorResult,
    TripPurposeResult,
    VisaCenterResult,
    VisaTypeResult,
    VisaValidityResult,
    VnjTypeResult,
)
from tqdm_utils import tqdm

load_dotenv()

RELEVANT_TAGS = ["#одобрено", "#одобрение", "#отказ"]

DEFAULT_MODEL = "google/gemma-4-31b-it:free"

OutputMode = Literal["tool", "prompted", "native"]
ProviderName = Literal["openrouter", "mistral"]


class ModelProviderConfig(BaseModel):
    output_type: OutputMode
    provider: ProviderName = "openrouter"


MODEL_PROVIDER_CONFIG: dict[str, ModelProviderConfig] = {
    "google/gemma-4-31b-it:free": ModelProviderConfig(output_type="native"),
    "deepseek/deepseek-v4-flash": ModelProviderConfig(output_type="prompted"),
    "mistral-small-latest": ModelProviderConfig(output_type="tool", provider="mistral"),
}


def build_output_type(
    model: str, schema: type[BaseModel], override: OutputMode | None
) -> ToolOutput[BaseModel] | PromptedOutput[BaseModel] | NativeOutput[BaseModel]:
    mode: OutputMode = override or MODEL_PROVIDER_CONFIG.get(model, ModelProviderConfig(output_type="tool")).output_type
    if mode == "tool":
        return ToolOutput(schema)
    elif mode == "prompted":
        return PromptedOutput(schema)
    return NativeOutput(schema)


class PromptConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_path: Path
    result_schema: type[BaseModel]
    results_dir: Path


PROMPT_REGISTRY: dict[str, PromptConfig] = {
    "dates": PromptConfig(
        prompt_path=Path("prompts/01_dates.txt"),
        result_schema=DatesResult,
        results_dir=Path("../data/extracted/01_dates"),
    ),
    "visa_type": PromptConfig(
        prompt_path=Path("prompts/02_visa_type.txt"),
        result_schema=VisaTypeResult,
        results_dir=Path("../data/extracted/02_visa_type"),
    ),
    "entries": PromptConfig(
        prompt_path=Path("prompts/03_entries.txt"),
        result_schema=EntriesResult,
        results_dir=Path("../data/extracted/03_entries"),
    ),
    "visa_validity": PromptConfig(
        prompt_path=Path("prompts/04_visa_validity.txt"),
        result_schema=VisaValidityResult,
        results_dir=Path("../data/extracted/04_visa_validity"),
    ),
    "employment": PromptConfig(
        prompt_path=Path("prompts/05_employment.txt"),
        result_schema=EmploymentResult,
        results_dir=Path("../data/extracted/05_employment"),
    ),
    "group": PromptConfig(
        prompt_path=Path("prompts/06_group.txt"),
        result_schema=GroupResult,
        results_dir=Path("../data/extracted/06_group"),
    ),
    "sponsor": PromptConfig(
        prompt_path=Path("prompts/07_sponsor.txt"),
        result_schema=SponsorResult,
        results_dir=Path("../data/extracted/07_sponsor"),
    ),
    "trip_purpose": PromptConfig(
        prompt_path=Path("prompts/08_trip_purpose.txt"),
        result_schema=TripPurposeResult,
        results_dir=Path("../data/extracted/08_trip_purpose"),
    ),
    "visa_center": PromptConfig(
        prompt_path=Path("prompts/09_visa_center.txt"),
        result_schema=VisaCenterResult,
        results_dir=Path("../data/extracted/09_visa_center"),
    ),
    "prior_schengen": PromptConfig(
        prompt_path=Path("prompts/10_prior_schengen.txt"),
        result_schema=PriorSchengenResult,
        results_dir=Path("../data/extracted/10_prior_schengen"),
    ),
    "rejection": PromptConfig(
        prompt_path=Path("prompts/11_rejection.txt"),
        result_schema=RejectionResult,
        results_dir=Path("../data/extracted/11_rejection"),
    ),
    "countries": PromptConfig(
        prompt_path=Path("prompts/12_countries.txt"),
        result_schema=CountriesResult,
        results_dir=Path("../data/extracted/12_countries"),
    ),
    "additional_docs": PromptConfig(
        prompt_path=Path("prompts/13_additional_docs.txt"),
        result_schema=AdditionalDocsResult,
        results_dir=Path("../data/extracted/13_additional_docs"),
    ),
    "vnj_type": PromptConfig(
        prompt_path=Path("prompts/14_vnj_type.txt"),
        result_schema=VnjTypeResult,
        results_dir=Path("../data/extracted/14_vnj_type"),
    ),
    "requested_vs_granted": PromptConfig(
        prompt_path=Path("prompts/15_requested_vs_granted.txt"),
        result_schema=RequestedVsGrantedResult,
        results_dir=Path("../data/extracted/15_requested_vs_granted"),
    ),
}


class RelevantMessage(BaseModel):
    id: int
    date: str
    author: str
    text: str
    tag: str
    year: str
    source_file: str


def get_tag(text: str) -> str | None:
    for tag in RELEVANT_TAGS:
        if tag in text:
            return tag
    return None


def country_from_path(filepath: Path) -> str:
    return filepath.parent.name


def load_processed_ids(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    ids: set[int] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["message_id"])
    return ids


def load_relevant_messages(filepath: Path) -> list[RelevantMessage]:
    if not filepath.exists():
        typer.echo(f"ERROR: {filepath} not found", err=True)
        raise typer.Exit(1)
    raw_data: dict = json.loads(filepath.read_text(encoding="utf-8"))
    file_year: str | None = str(raw_data["year"]) if "year" in raw_data else None
    messages: list[RelevantMessage] = []
    for raw_msg in raw_data["messages"]:
        text = raw_msg.get("text", "")
        if not isinstance(text, str):
            continue
        tag = get_tag(text)
        if tag is None:
            continue
        year = file_year if file_year is not None else str(datetime.fromisoformat(raw_msg["date"]).year)
        messages.append(
            RelevantMessage(
                id=raw_msg["id"],
                date=raw_msg["date"],
                author=raw_msg["from"],
                text=text,
                tag=tag,
                year=year,
                source_file=filepath.name,
            )
        )
    return messages


async def process_message(
    msg: RelevantMessage,
    agent: Agent[None, BaseModel],
    prompt_template: str,
    semaphore: asyncio.Semaphore,
    limiter: AsyncLimiter | None,
    output_path: Path,
    lock: asyncio.Lock,
    country: str,
    model_name: str,
    processed_ids: set[int],
) -> None:
    if msg.id in processed_ids:
        return
    processed_ids.add(msg.id)

    prompt_text = prompt_template.replace("{YEAR}", msg.year).replace("{MESSAGE}", msg.text)

    retry_run = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(json.JSONDecodeError),
    )(agent.run)

    async with semaphore:
        if limiter is not None:
            async with limiter:
                result = await retry_run(prompt_text)
        else:
            result = await retry_run(prompt_text)

    usage = result.usage()
    cost_usd: float = sum(
        m.provider_details["cost"]
        for m in result.all_messages()
        if isinstance(m, ModelResponse)
        and isinstance(m.provider_details, dict)
        and "cost" in m.provider_details
    )

    record = {
        "country": country,
        "model": model_name,
        "message_id": msg.id,
        "source_file": msg.source_file,
        "message_date": msg.date,
        "author": msg.author,
        "tag": msg.tag,
        "extracted": result.output.model_dump(),
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "requests": usage.requests,
            "cost_usd": cost_usd,
        },
        "message_text": msg.text,
    }

    async with lock:
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_model(model_name: str, provider_override: ProviderName | None) -> tuple[OpenRouterModel | MistralModel, OpenRouterModelSettings | None]:
    provider: ProviderName = provider_override or MODEL_PROVIDER_CONFIG.get(model_name, ModelProviderConfig(output_type="tool")).provider
    if provider == "mistral":
        return MistralModel(model_name, provider=MistralProvider(api_key=os.environ["MISTRAL_API_KEY"])), None
    return OpenRouterModel(model_name, provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"])), OpenRouterModelSettings(openrouter_reasoning={"effort": "high"})


async def run_async(
    input_file: Path,
    prompt_name: str,
    model_name: str,
    limit: int | None,
    concurrency: int,
    rpm: int | None,
    output_mode: OutputMode | None,
    provider_override: ProviderName | None,
) -> None:
    config = PROMPT_REGISTRY[prompt_name]
    config.results_dir.mkdir(parents=True, exist_ok=True)
    prompt_template = config.prompt_path.read_text(encoding="utf-8")

    country = country_from_path(input_file)
    output_path = config.results_dir / f"{country}.jsonl"
    processed_ids = load_processed_ids(output_path)

    model, settings = build_model(model_name, provider_override)
    agent: Agent[None, BaseModel] = Agent(
        model,
        output_type=build_output_type(model_name, config.result_schema, output_mode),
        **({"model_settings": settings} if settings else {}),
    )

    all_messages = load_relevant_messages(input_file)
    total_relevant = len(all_messages)

    if limit is not None:
        all_messages = all_messages[:limit]

    typer.echo(f"Found {total_relevant} relevant messages total, will process {len(all_messages)}")

    semaphore = asyncio.Semaphore(concurrency)
    limiter = AsyncLimiter(rpm, 60) if rpm is not None else None
    lock = asyncio.Lock()

    async def process_with_progress(msg: RelevantMessage, pbar: tqdm) -> None:
        await process_message(msg, agent, prompt_template, semaphore, limiter, output_path, lock, country, model_name, processed_ids)
        pbar.update(1)

    with tqdm(total=len(all_messages), desc="Processing messages") as pbar:
        tasks = [process_with_progress(msg, pbar) for msg in all_messages]
        await asyncio.gather(*tasks)

    typer.echo(f"Done. Output: {output_path} ({len(processed_ids)} total records)")


app = typer.Typer()

PROMPT_NAMES = ", ".join(PROMPT_REGISTRY)


@app.command()
def run(
    input_file: Path = typer.Argument(help="Path to input clean JSON file"),
    prompt: str = typer.Option(..., help=f"Prompt name: {PROMPT_NAMES}"),
    model: str = typer.Option(DEFAULT_MODEL, help="Model name"),
    limit: int | None = typer.Option(None, help="Maximum number of messages to process"),
    concurrency: int = typer.Option(5, help="Number of concurrent requests"),
    rpm: int | None = typer.Option(None, help="Maximum requests per minute"),
    output_type: OutputMode | None = typer.Option(None, help="Output mode: tool, prompted, native (default: from MODEL_PROVIDER_CONFIG)"),
    provider: ProviderName | None = typer.Option(None, help="Provider: openrouter, mistral (default: from MODEL_PROVIDER_CONFIG)"),
) -> None:
    if prompt not in PROMPT_REGISTRY:
        typer.echo(f"Unknown prompt '{prompt}'. Available: {PROMPT_NAMES}", err=True)
        raise typer.Exit(1)
    asyncio.run(run_async(input_file=input_file, prompt_name=prompt, model_name=model, limit=limit, concurrency=concurrency, rpm=rpm, output_mode=output_type, provider_override=provider))


@app.command()
def show_samples(
    input_file: Path = typer.Argument(help="Path to input clean JSON file"),
    n: int = typer.Option(10, help="Number of messages to show"),
) -> None:
    messages = load_relevant_messages(input_file)
    for msg in messages[:n]:
        typer.echo(f"\n--- ID={msg.id} | {msg.date} | {msg.author} | {msg.tag} ---")
        typer.echo(msg.text[:300])



if __name__ == "__main__":
    app()
