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
    chunk_length: int = 1000 # max chunk length (characters)
    chunk_overlap: int = 200 # max chunk overlap between 2 consecutive chunks (characters)
    chunk_size: int = 32 # max chunk number per batch (must be <32 with BAAI/bge-m3 + only-cpu)

    # Embedding openAI
    # embedding_model: str = "text-embedding-3-small" 
    # model_dimensions: int = 1536
    # par défaut text-embedding-3-small: dimensions=1536
    
    # Embedding hugging-face
    embedding_model: str = "BAAI/bge-m3"
    embedding_api_url: str = "http://localhost:8080"
   

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