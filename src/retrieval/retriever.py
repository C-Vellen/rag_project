from langchain_postgres import PGVector
from langchain_core.vectorstores import VectorStoreRetriever
from ..ingestion.embedder import get_vectorstore
from ..config import settings

def get_retriever(
    search_type=settings.search_type,
    k=settings.k,
    score_threshold=settings.score_threshold,
) -> VectorStoreRetriever:
    """
    Retourne un retriever branché sur le vectorstore pgvector.

    search_type:
        - "similarity"           : top-k par similarité cosinus (défaut)
        - "mmr"                  : Maximum Marginal Relevance (diversité)
        - "similarity_score_threshold" : filtre par score minimum
    """
    vectorstore: PGVector = get_vectorstore()

    search_kwargs: dict = {"k": k}

    if search_type == "similarity_score_threshold":
        if score_threshold is None:
            score_threshold = 0.7
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    return retriever
