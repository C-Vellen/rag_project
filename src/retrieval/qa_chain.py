from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from ..retrieval.retriever import get_retriever
from ..ingestion.embedder import get_vectorstore
from ..config import settings


PROMPT_TEMPLATE = """Tu es un assistant expert. Réponds à la question en te basant \
uniquement sur le contexte fourni. Si la réponse ne s'y trouve pas, dis-le clairement.

Contexte :
{context}

Question : {question}
"""


def fetch_chunks_with_scores(question: str, k: int = 4) -> list[tuple]:
    """Récupère les chunks avec leur score de similarité."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(question, k=k)


def display_full_prompt(question: str, k: int = 4) -> str:
    """Affiche le prompt complet en console et retourne le contexte formaté."""
    chunks_with_scores = fetch_chunks_with_scores(question, k=k)

    lines = []
    lines.append("\n" + "═" * 70)
    lines.append("📋 PROMPT COMPLET ENVOYÉ AU LLM")
    lines.append("═" * 70)

    # 1. Prompt template
    lines.append("\n📌 PROMPT TEMPLATE :")
    lines.append("-" * 40)
    lines.append(PROMPT_TEMPLATE)

    # 2. Chunks avec scores
    lines.append("📚 CONTEXTE — chunks sélectionnés :")
    lines.append("-" * 40)
    context_parts = []
    for i, (doc, score) in enumerate(chunks_with_scores):
        source = doc.metadata.get("source", "inconnue")
        lines.append(f"\n[Chunk {i+1}]")
        lines.append(f"  📁 Source    : {source}")
        lines.append(f"  📐 Distance  : {score:.4f}")
        
    print("\n".join(lines))


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
    retriever = get_retriever(search_type="similarity", k=4) 

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # # print(f"- Contexte général: \n{prompt}\n- Contexte trouvé dans les documents:\n{format_docs(retriever)}\n")
    # print("-----------------------------------------------------------------------")
    # print("Prompt envoyé au chatbot: ")
    # print(PROMPT_TEMPLATE.format(context="retrieval", question="question??" ) 
    # print("-----------------------------------------------------------------------")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
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