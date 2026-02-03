import oci
import os
from typing import Any, Dict, List
from app.services.oci_llm import call_oci_chat
from app.services.embedding_service import OCIEmbeddingService
from app.services.vector_store_service import search_similar_chunks
from app.services.secure_config import require_env


# ------ VM Configuration -------
# config = oci.config.from_file(
#     file_location="/home/opc/.oci/config",
#     profile_name="GC3TEST02"
# )


# ---- OCI Configuration ----
CONFIG_PROFILE = require_env("CONFIG_PROFILE")
config = oci.config.from_file(
    file_location=r"C:\\Users\\shshrohi\\.oci\\config",
    profile_name=CONFIG_PROFILE
)
session_history: Dict[str, list] = {}

compartment_id = require_env("COMPARTMENT_ID")
MODEL_ID = require_env("MODEL_ID")
endpoint = require_env("ENDPOINT")

generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config=config,
    service_endpoint=endpoint,
    retry_strategy=oci.retry.NoneRetryStrategy(),
    timeout=(10, 240)
)

def ai_redact_sensitive_info(text: str) -> str:
    USER_MESSAGE = f"""
        You are a data anonymization expert. Replace all company/customer names or any critical information
        that are NOT 'Oracle' or similar with [Anonymized Customer].

        Return only the anonymized text.

        Original Text: {text}

        Anonymized Text:
    """

    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    chat_request = oci.generative_ai_inference.models.CohereChatRequest()

    chat_request.message = USER_MESSAGE
    chat_request.max_tokens = 4000
    chat_request.temperature = 1

    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=MODEL_ID)
    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = compartment_id

    response = generative_ai_inference_client.chat(chat_detail)
    return response.data.chat_response.text

# MAIN RAG + CONVERSATION FUNCTION

def answer_query(query: str, top_k: int = 5, session_id: str | None = None) -> Dict[str, Any]:
    global session_history

    if not session_id:
        session_id = "default-session"

    if session_id not in session_history:
        session_history[session_id] = []

    history = session_history[session_id]

    embedder = OCIEmbeddingService()
    query_embedding = embedder.embed_text(query)

    hits = search_similar_chunks(query_embedding, top_k=top_k)

    documents = []
    for i, h in enumerate(hits):
        documents.append({
            "title": f"Document Chunk {i+1}",
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
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": llm_output})

    return {
        "answer": llm_output,
        "chunks": hits,
        "history_length": len(history),
        "session_id": session_id
    }