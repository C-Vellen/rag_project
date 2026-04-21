import sys
from pathlib import Path
from .ingestion.loader import load_documents
from .ingestion.splitter import split_documents
from .ingestion.embedder import embed_and_store, embed_dryrun
from .retrieval.qa_chain import get_qa_chain, display_full_prompt
from .config import settings

def ingest(source: str | Path) -> None:
    """Pipeline complète d'ingestion : load → split → embed → store."""
    print("\n============= Ingestion ==================")
    print(f"\n📂 Chargement des documents depuis : {source}")
    documents = load_documents(source)
    print(f"  → {len(documents)} document(s) chargé(s)")

    print("\n✂️  Découpage en chunks...")
    chunks = split_documents(documents)

    print("\n🔢 Embedding et stockage dans pgvector...")
    embed_and_store(chunks)
    # embed_dryrun(chunks)

    print("\n✅ Ingestion terminée !\n")

def query(question: str) -> None:
    """Pipeline de retrieval + génération."""


    print(f"\n🔍 Question : {question}\n")
    chain = get_qa_chain()

    if settings.debug:
        display_full_prompt(question)

    # Streaming de la réponse
    print("\n" + "═" * 70)
    print("💬 Réponse :")
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    
    if command == "ingest":
        source = sys.argv[2] if len(sys.argv) > 2 else "src/documents/"
        ingest(source)

    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage : poetry run python -m rag_project.main query 'ta question'")
            sys.exit(1)
        query(sys.argv[2])

    else:
        print("Usage:")
        print("  poetry run python -m rag_project.main ingest <dossier|fichier>")
        print("  poetry run python -m rag_project.main query 'ta question'")
        
        
    # source = sys.argv[1] if len(sys.argv) > 1 else "documents/"
    # ingest(source)