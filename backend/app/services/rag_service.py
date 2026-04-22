from typing import Any, Dict, List
from app.services.oci_llm import call_oci_chat, call_oci_chat_generic, call_oci_chat_maverick, call_oci_title
from app.services.embedding_service import OCIEmbeddingService
from app.services.retrieval_service import search_similar_chunks
from app.services.secure_config import get_env


HISTORY_TURNS = int(get_env("HISTORY_TURNS", "20"))
DEFAULT_CHAT_MODEL = "cohere"
SUPPORTED_CHAT_MODELS = {"cohere", "maverick", "gpt-5.2"}
DEFAULT_TOP_K = int(get_env("DEFAULT_TOP_K", "12"))

def _limit_title_words(title: str, max_words: int = 5) -> str:
    words = title.strip().split()
    return " ".join(words[:max_words])


def _derive_session_title(query: str, max_words: int = 5) -> str:
    title = " ".join(query.strip().split())
    return _limit_title_words(title, max_words=max_words)


def build_citations_from_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for idx, hit in enumerate(hits, start=1):
        metadata = (hit or {}).get("metadata", {}) or {}
        citations.append(
            {
                "rank": idx,
                "chunk_id": metadata.get("chunk_id") or f"chunk_{idx}",
                "source_file": metadata.get("source_file"),
                "chunk_index": metadata.get("chunk_index"),
                "heading": metadata.get("heading"),
            }
        )
    return citations

# MAIN RAG + CONVERSATION FUNCTION

def answer_query(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    history: List[Dict[str, str]] | None = None,
    model: str = DEFAULT_CHAT_MODEL,
    generate_title: bool = False,
) -> Dict[str, Any]:
    model_key = (model or DEFAULT_CHAT_MODEL).strip().lower()
    if model_key not in SUPPORTED_CHAT_MODELS:
        model_key = DEFAULT_CHAT_MODEL

    incoming_history = history or []
    if HISTORY_TURNS > 0:
        # Keep only the latest N turns to reduce context size.
        incoming_history = incoming_history[-HISTORY_TURNS:]

    embedder = OCIEmbeddingService()
    query_embedding = embedder.embed_text(query)
    if not query_embedding:
        raise ValueError("Failed to generate embedding for the query.")

    hits = search_similar_chunks(query_embedding, top_k=top_k)

    documents = []
    for i, h in enumerate(hits):
        metadata = h.get("metadata", {}) or {}
        source_file = metadata.get("source_file") or f"Document_{i+1}.docx"
        heading = metadata.get("heading") or "[no heading]"
        sheet_name = metadata.get("sheet_name")
        row_start = metadata.get("row_start")
        row_end = metadata.get("row_end")
        row_range = None
        if row_start is not None and row_end is not None:
            row_range = f"Rows: {row_start}-{row_end}"
        elif metadata.get("row_number") is not None:
            row_range = f"Row: {metadata.get('row_number')}"

        context_parts = [f"Source: {source_file}", f"Heading: {heading}"]
        if sheet_name:
            context_parts.append(f"Sheet: {sheet_name}")
        if row_range:
            context_parts.append(row_range)

        documents.append({
            "title": source_file,
            "snippet": " | ".join(context_parts) + "\n" + h["chunk"]
        })

    cohere_history = []
    for turn in incoming_history:
        role = (turn or {}).get("role", "")
        content = (turn or {}).get("content", "")
        if not content:
            continue
        role_lower = role.lower()
        if role_lower == "user":
            cohere_history.append({
                "role": "USER",
                "message": content
            })
        elif role_lower == "assistant":
            cohere_history.append({
                "role": "CHATBOT",
                "message": content
            })

    if not documents:
        documents = [{
            "title": "No matching knowledge found",
            "snippet": "No relevant knowledge base chunks were found for this query. Respond carefully and mention that no strong source context was retrieved.",
        }]

    # ---- Call selected chat model ----
    if model_key == "maverick":
        llm_output = call_oci_chat_maverick(
            message=query,
            chat_history=cohere_history,
            documents=documents,
        )
    elif model_key == "gpt-5.2":
        llm_output = call_oci_chat_generic(
            message=query,
            chat_history=cohere_history,
            documents=documents,
        )
    else:
        llm_output = call_oci_chat(
            message=query,
            chat_history=cohere_history,
            documents=documents,
        )
    generated_title = None
    if generate_title and query:
        try:
            generated_title = call_oci_title(query, max_words=5)
        except Exception:
            generated_title = _derive_session_title(query, max_words=5)
        generated_title = _limit_title_words(generated_title, max_words=5)

    return {
        "answer": llm_output,
        "chunks": hits,
        "citations": build_citations_from_hits(hits),
        "history_length": len(incoming_history),
        "model_used": model_key,
        "generated_title": generated_title,
    }
