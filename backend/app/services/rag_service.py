from typing import Any, Dict, List
from app.services.oci_llm import call_oci_chat, call_oci_title
from app.services.embedding_service import OCIEmbeddingService
from app.services.vector_store_service import search_similar_chunks
from app.services.secure_config import get_env
from app.services.session_store_service import (
    fetch_session_history,
    insert_session_message,
    set_session_title_if_empty,
)


HISTORY_TURNS = int(get_env("HISTORY_TURNS", "20"))

def _limit_title_words(title: str, max_words: int = 5) -> str:
    words = title.strip().split()
    return " ".join(words[:max_words])


def _derive_session_title(query: str, max_words: int = 5) -> str:
    title = " ".join(query.strip().split())
    return _limit_title_words(title, max_words=max_words)

# MAIN RAG + CONVERSATION FUNCTION

def answer_query(query: str, top_k: int = 5, session_id: str | None = None) -> Dict[str, Any]:
    if not session_id:
        session_id = "default-session"

    history = fetch_session_history(session_id, limit=HISTORY_TURNS)

    embedder = OCIEmbeddingService()
    query_embedding = embedder.embed_text(query)

    hits = search_similar_chunks(query_embedding, top_k=top_k)

    documents = []
    for i, h in enumerate(hits):
        metadata = h.get("metadata", {}) or {}
        source_file = metadata.get("source_file") or f"Document_{i+1}.docx"
        documents.append({
            "title": source_file,
            "snippet": h["chunk"]
        })

    cohere_history = []
    for turn in history:
        if turn["role"] == "user":
            cohere_history.append({
                "role": "USER",
                "message": turn["content"]
            })
        elif turn["role"] == "assistant":
            cohere_history.append({
                "role": "CHATBOT",
                "message": turn["content"]
            })

    # ---- Call OCI Cohere Chat ----
    llm_output = call_oci_chat(
        message=query,
        chat_history=cohere_history,
        documents=documents
    )

    # ---- Persist history ----
    insert_session_message(session_id, "user", query)
    insert_session_message(session_id, "assistant", llm_output)
    if query:
        try:
            title = call_oci_title(query, max_words=5)
        except Exception:
            title = _derive_session_title(query, max_words=5)
        title = _limit_title_words(title, max_words=5)
        if title:
            set_session_title_if_empty(session_id, title)

    return {
        "answer": llm_output,
        "chunks": hits,
        "history_length": len(history) + 2,
        "session_id": session_id
    }
