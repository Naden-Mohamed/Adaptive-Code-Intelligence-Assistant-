from __future__ import annotations 
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = ""
    model_id: str = "openai/gpt-oss-120b"
    embedding_model: str = "BAAI/bge-m3"
    temperature : float = 0.0

    # Vector store
    vector_db: str = "chroma"
    vector_db_path: str = "./vectorstore/index"
    top_k: int = 4
    similarity_threshold: float = 0.95
    chunk_size: int = 1500
    chunk_overlap: int = 150

    # Dataset
    dataset_path: str = 'data/human-eval-v2-20210705.jsonl'

    # Router
    router_min_confidence: float = 0.55

    # Sandbox execution
    sandbox_timeout_seconds: int = 8
    sandbox_memory_limit_mb: int = 256

    # Observability
    log_dir: str = "./logs"

    # Resilience
    max_retries: int = 3
    request_timeout_seconds: int = 30

#stands for Least Recently Used Cache and is a decorator that stores a function's results 
#so that the same results can be reused if the function is called with the same arguments
@lru_cache
def get_settings()-> Settings:
    return Settings()