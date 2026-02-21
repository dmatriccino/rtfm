"""Application settings with environment variable support."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """RTFM configuration settings."""

    model_config = {"env_prefix": "RTFM_", "env_file": ".env", "env_file_encoding": "utf-8"}

    # Paths
    data_dir: Path = Path("data")
    books_dir: Path = Path("data/books")
    chroma_db_dir: Path = Path("data/chroma_db")

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    # Chunking
    max_chunk_size: int = 1500
    max_code_chunk_size: int = 3000
    chunk_overlap: int = 200

    # Retrieval
    default_top_k: int = 5
    default_collection: str = "engineering"

    # Logging
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()
