"""
Configuration settings for ChromaDB Service
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # Service Configuration
    DEBUG: bool = False
    SERVICE_NAME: str = "ChromaDB Service"

    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_data"
    CHROMA_COLLECTION_NAME: str = "database_registry"

    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Server Configuration
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = True


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()


settings = get_settings()