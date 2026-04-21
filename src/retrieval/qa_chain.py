import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from ..retrieval.retriever import get_retriever
from ..ingestion.embedder import get_vectorstore
from ..config import settings


PROMPT_TEMPLATE = """Tu es un assistant expert. Réponds à la question en te basant \
uniquement sur le contexte fourni. Si la réponse ne s'y trouve pas, dis-le clairement.

Contexte :
{context}

Question : {question}
"""


def fetch_chunks_with_scores(question: str, k: int = settings.k) -> list[tuple]:
    """Récupère les chunks avec leur score de similarité."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(question, k=k)


def display_full_prompt(question: str, k: int = settings.k) -> str:
    """Affiche le prompt complet en console et retourne le contexte formaté."""
    
    vectorstore = get_vectorstore() 
    chunks_with_scores = vectorstore.similarity_search_with_score(question, k=k)

    # 1. Vectorisation de la question
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.model_dimensions,
        api_key=settings.openai_api_key,
    )
    q_vector = embeddings.embed_query(question)
    print("➡️ Vectorisation de la question :")
    print("\tVecteur | Dimension |      Norme | Vecteur")
    print(f"\tQuestion| {len(q_vector):>9} | {np.linalg.norm(q_vector):>10.7f} | [{','.join(f'{v:4f}' for v in q_vector[:5])}, ...]")
    
    # 2. Chunks séléctionnés
    print("\n📚 RETRIVAL chunks sélectionnés :")
    print("\tChunk| Distance |    Vecteur                        | Document source                          | Text")
    for i, (doc, score) in enumerate(chunks_with_scores):
        doc_source = doc.metadata.get("source", "inconnue").split('/')[-1]
        text = doc.page_content
        print(f"\t {i+1:>3} | {score:>8.4f} | [{','.join(f'{v:4f}' for v in q_vector[:3])}, ...] | {doc_source[:40]} | {text[:60]}...")
              
    # 3. Prompt envoyé au LLM
    print("\n📋 PROMPT COMPLET ENVOYÉ AU LLM\n")
    print(PROMPT_TEMPLATE.format(
        context="\n".join([doc[0].page_content for doc in chunks_with_scores ]), 
        question=question
        ))
    

def format_docs(docs) -> str:
    """Concatène les chunks récupérés en un seul bloc de contexte."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'inconnue')}]\n{doc.page_content}"
        for doc in docs
    )


def get_qa_chain():
    """
    Construit la chaîne RAG complète :
    question → retriever → prompt → LLM → réponse
    """
    retriever = get_retriever(search_type="similarity", k=settings.k) 

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )

    # Chaîne LCEL (LangChain Expression Language)
    chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain