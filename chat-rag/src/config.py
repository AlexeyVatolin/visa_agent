from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mistral_api_key: str
    data_path: str = "data/messages.json"
    official_data_path: str = "data/germany_visa_official.json"
    chroma_path: str = "chroma_db"
    collection_name: str = "chat_history"

    # LangSmith tracing — set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable
    langsmith_tracing: bool = False
    langsmith_api_key: str
    langsmith_project: str
    langsmith_endpoint: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

settings = Settings()
