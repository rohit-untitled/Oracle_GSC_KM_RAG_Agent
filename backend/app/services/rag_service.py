from typing import Any, Dict, List, Optional
from app.services.oci_llm import call_oci_chat, call_oci_chat_generic, call_oci_chat_maverick, call_oci_title
from app.services.embedding_service import OCIEmbeddingService
from app.services.retrieval_service import search_similar_chunks
from app.services.secure_config import get_env


HISTORY_TURNS = int(get_env("HISTORY_TURNS", "15"))
DEFAULT_CHAT_MODEL = "cohere"
SUPPORTED_CHAT_MODELS = {"cohere", "maverick", "gpt-5.2"}
DEFAULT_TOP_K = int(get_env("DEFAULT_TOP_K", "8"))
SUPPORTED_CONFIDENTIALITY = {"SCM", "ERP", "EPM"}


RETRIEVAL_PROFILES = {
    "instant": {
        "top_k": 3,
        "rerank_top_n": 4,
        "neighbor_radius": 0,
        "use_hybrid": False,
    },
    "thinking": {
        "top_k": 8,
        "rerank_top_n": 12,
        "neighbor_radius": 1,
        "use_hybrid": True,
    },
    "pro": {
        "top_k": 12,
        "rerank_top_n": 18,
        "neighbor_radius": 2,
        "use_hybrid": True,
    },
}

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


def _resolve_retrieval_profile(mode: str | None, top_k_override: int | None) -> Dict[str, Any]:
    mode_key = (mode or "thinking").strip().lower()
    if mode_key not in RETRIEVAL_PROFILES:
        mode_key = "thinking"

    profile = dict(RETRIEVAL_PROFILES[mode_key])
    if top_k_override is not None:
        profile["top_k"] = top_k_override
    profile["mode"] = mode_key
    return profile

# MAIN RAG + CONVERSATION FUNCTION

def answer_query(
    query: str,
    top_k: int | None = None,
    history: List[Dict[str, str]] | None = None,
    model: str = DEFAULT_CHAT_MODEL,
    generate_title: bool = False,
    mode: str = "thinking",
    confidentiality: str = "SCM",
) -> Dict[str, Any]:
    model_key = (model or DEFAULT_CHAT_MODEL).strip().lower()
    if model_key not in SUPPORTED_CHAT_MODELS:
        model_key = DEFAULT_CHAT_MODEL

    incoming_history = history or []
    if HISTORY_TURNS > 0:
        # Keep only the latest N turns to reduce context size.
        incoming_history = incoming_history[-HISTORY_TURNS:]

    confidentiality_key = (confidentiality or "SCM").strip().upper()
    if confidentiality_key not in SUPPORTED_CONFIDENTIALITY:
        confidentiality_key = "SCM"

    embedder = OCIEmbeddingService()
    query_embedding = embedder.embed_text(query)
    if not query_embedding:
        raise ValueError("Failed to generate embedding for the query.")

    retrieval_profile = _resolve_retrieval_profile(mode, top_k)

    hits = search_similar_chunks(
        query_embedding,
        top_k=retrieval_profile["top_k"],
        query_text=query,
        rerank_top_n=retrieval_profile["rerank_top_n"],
        neighbor_radius=retrieval_profile["neighbor_radius"],
        use_hybrid=retrieval_profile["use_hybrid"],
        confidentiality=confidentiality_key,
    )

    if not hits and retrieval_profile["mode"] == "instant":
        hits = search_similar_chunks(
            query_embedding,
            top_k=6,
            query_text=query,
            rerank_top_n=10,
            neighbor_radius=1,
            use_hybrid=True,
            confidentiality=confidentiality_key,
        )

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
            "snippet": "I couldn’t find a matching project document for this question. I’ll answer using general Oracle Fusion guidance, and I’ll call out what would need confirmation from the relevant setup workbook, mapping file, design document, or implementation document.",
        }]

    # ---- Call selected chat model ----
    if model_key == "maverick":
        llm_output = call_oci_chat_maverick(
            message=query,
            chat_history=cohere_history,
            documents=documents,
            confidentiality=confidentiality_key,
        )
    elif model_key == "gpt-5.2":
        llm_output = call_oci_chat_generic(
            message=query,
            chat_history=cohere_history,
            documents=documents,
            confidentiality=confidentiality_key,
        )
    else:
        llm_output = call_oci_chat(
            message=query,
            chat_history=cohere_history,
            documents=documents,
            confidentiality=confidentiality_key,
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
        "confidentiality": confidentiality_key,
        "generated_title": generated_title,
        "retrieval_config": {
            "mode": retrieval_profile["mode"],
            "top_k": retrieval_profile["top_k"],
            "rerank_top_n": retrieval_profile["rerank_top_n"],
            "neighbor_radius": retrieval_profile["neighbor_radius"],
            "use_hybrid": retrieval_profile["use_hybrid"],
            "confidentiality": confidentiality_key,
        },
    }
