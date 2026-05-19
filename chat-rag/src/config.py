from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).parent.parent  # chat-rag/


class Settings(BaseSettings):
    mistral_api_key: str
    gemini_api_key: str | None = None
    data_path: Path = _ROOT / "data/messages.json"
    official_data_path: Path = _ROOT / "data/germany_visa_official.json"
    chroma_path: Path = _ROOT / "chroma_db"
    collection_name: str = "chat_history"

    # LangSmith tracing — set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable
    langsmith_tracing: bool = False
    langsmith_api_key: str
    langsmith_project: str
    langsmith_endpoint: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")


settings = Settings()
