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
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embedding 
    embedding_model: str = "text-embedding-3-small" 
    model_dimensions: int = 1536
    # par défaut text-embedding-3-small: dimensions=1536

    # pgvector
    collection_name: str = "documents"
    
    # retrieval
    search_type: str = "similarity" 
    """
    search_type:
        - "similarity"           : top-k par similarité cosinus (défaut)
        - "mmr"                  : Maximum Marginal Relevance (diversité)
        - "similarity_score_threshold" : filtre par score minimum
    """
    score_threshold: float | None = None
    k: int = 4 # nombre de chunks sélectionnés
    
    #llm
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    
    
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