from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model_name:  str = "mistral:latest"
    llm_temp: float = Field(0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field (2048, gt=0)
    llm_timeout: int = Field(120, gt=0)

    data_dir: Path = Path("./data")
    chroma_db_dir: Path = Path("./chroma_db")
    sqlite_db_path: Path = Path("./db/brainy_binder.db")

    embedding_model_name: str = "all-MiniLM-L6-v2"
    top_k: int = Field(5, gt=0)
    chunk_size: int = Field(1000, gt=0) # Max size of chunks in charactors
    chunk_overlap: int = Field(300, ge=0)
    chroma_collection_name: str = "brainy_binder"

settings = Settings()