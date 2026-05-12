from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mistral_api_key: str
    data_path: str = "data/messages.json"
    chroma_path: str = "chroma_db"
    collection_name: str = "chat_history"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
