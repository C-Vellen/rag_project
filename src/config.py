from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str
    database_url: str

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embedding
    embedding_model: str = "text-embedding-3-small"

    # pgvector
    collection_name: str = "documents"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"  # ← ignore les variables d'env inconnues
        )

settings = Settings()