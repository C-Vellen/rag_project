from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from ..config import settings

def get_vectorstore() -> PGVector:
    """Retourne le vectorstore connecté à PostgreSQL."""
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return PGVector(
        embeddings=embeddings,
        # collection_name=settings.collection_name,
        collection_name="documents",
        connection=settings.database_url,
        # Crée la table + l'extension pgvector si elle n'existe pas
        create_extension=True,
    )

def embed_and_store(chunks: list[Document]) -> PGVector:
    """Calcule les embeddings et stocke les chunks dans pgvector."""
    print(f"  → Calcul des embeddings pour {len(chunks)} chunks...")
    print(repr(settings.database_url))
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print("  → Chunks stockés dans PostgreSQL ✓")
    return vectorstore