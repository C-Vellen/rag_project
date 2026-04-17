from pathlib import Path
from .ingestion.loader import load_documents
from .ingestion.splitter import split_documents
from .ingestion.embedder import embed_and_store, embed_dryrun

def ingest(source: str | Path) -> None:
    """Pipeline complète d'ingestion : load → split → embed → store."""
    print(f"\n📂 Chargement des documents depuis : {source}")
    documents = load_documents(source)
    print(f"  → {len(documents)} document(s) chargé(s)")

    print("\n✂️  Découpage en chunks...")
    chunks = split_documents(documents)

    print("\n🔢 Embedding et stockage dans pgvector...")
    embed_and_store(chunks)
    # embed_dryrun(chunks)

    print("\n✅ Ingestion terminée !")

if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else "documents/"
    ingest(source)