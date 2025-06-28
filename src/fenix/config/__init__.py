import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Google AI Configuration
    google_api_key: str
    gemini_model: str = "models/embedding-001"

    # Database Configuration
    database_url: str = "postgresql+psycopg2://postgres:password@localhost:5432/gemini_docs_db"
    async_database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/gemini_docs_db"

    # Vector Configuration
    vector_dimension: int = 768
    collection_name: str = "gemini_documents"

    # Performance Settings
    max_batch_size: int = 100
    connection_pool_size: int = 10

    # Application Settings
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False

    def validate_api_key(self) -> bool:
        """Validate that the API key is set and not the placeholder"""
        return (
            self.google_api_key
            and self.google_api_key != "your_google_api_key_here"
            and len(self.google_api_key) > 10
        )


# Global settings instance
settings = Settings()
