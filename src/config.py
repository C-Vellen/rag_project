from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field

class Settings(BaseSettings):
    
    # .env
    openai_api_key: str
    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: str
    debug: bool
       
    # Chunking
    chunk_size: int = 100
    chunk_overlap: int = 20

    # Embedding 
    embedding_model: str = "text-embedding-3-small" 
    model_dimensions: int = 512
    # par défaut text-embedding-3-small: dimensions=1536

    # pgvector
    collection_name: str = "documents"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  
        )
    
    @computed_field
    @property
    def database_url(self) -> str:
        return(
            f"postgresql+psycopg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        )

settings = Settings()