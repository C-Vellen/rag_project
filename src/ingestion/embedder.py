import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from ..config import settings

def get_vectorstore_explicit(chunks):
    """Retourne les vecteurs en permettant leur capture pour debug"""
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.model_dimensions,
        api_key=settings.openai_api_key,
    )
    texts = [doc.page_content for doc in chunks]
    metadatas=[d.metadata for d in chunks]
    
    # Créer les vecteurs explicitement
    vectors = embeddings.embed_documents(texts)
    print("> 5 premiers Embeddings:---------------------------------")
    print("Vecteur | Dimension |      Norme | 5 premières coordonnées")
    for i, vec in enumerate(vectors[:5]):
        print(f"    {i:03d} | {len(vec):>9} | {np.linalg.norm(vec):>10.7f} | {vec[:5]}")
    
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=settings.collection_name,
        connection=settings.database_url,
        # Crée la table + l'extension pgvector si elle n'existe pas
        create_extension=True,
    )
    # Enregistre les données dans la table:
    vectorstore.add_embeddings(
        texts=texts,
        embeddings=vectors,
        metadatas=metadatas
    )
    
    return texts, vectors



def get_vectorstore() -> PGVector:
    """Retourne le vectorstore connecté à PostgreSQL."""
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.model_dimensions,
        api_key=settings.openai_api_key,
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=settings.collection_name,
        connection=settings.database_url,
        # Crée la table + l'extension pgvector si elle n'existe pas
        create_extension=True,
    )

def embed_and_store(chunks: list[Document]) -> PGVector:
    """Calcule les embeddings et stocke les chunks dans pgvector."""
    print(f"  → Calcul des embeddings pour {len(chunks)} chunks...")
    print(f" Préparation du chargement dans la BD: {repr(settings.database_url)}")
    print(f"Debug: {settings.debug}")
    if settings.debug:    
        # Option debuggage: capture des embeddings avant chargement dans la BD
        vectorstore = get_vectorstore_explicit(chunks)
    else:       
        # Option production: chargement direct des embeddings dans la BD
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
    
    print("  → Chunks stockés dans PostgreSQL ✓")
    return vectorstore

def embed_dryrun(chunks: list[Document]):
    """Dryrun embeddings """
    print(f"  → Calcul des embeddings pour {len(chunks)} chunks...")
    print(repr(settings.database_url))
    print("  → Chunks non stockés dans PostgreSQL ✓")
    