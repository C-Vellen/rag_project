from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..config import settings

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Découpe les documents en chunks avec RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        # Séparateurs par ordre de priorité
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,   # Ajoute la position d'origine dans les métadonnées
    )
    chunks = splitter.split_documents(documents)
    print(f"  → {len(documents)} document(s) découpés en {len(chunks)} chunks")
    print("> 5 premiers Chunks:--------------------------------------------------------- ")
    for chunk in chunks[:5]:
        print(chunk)
    return chunks