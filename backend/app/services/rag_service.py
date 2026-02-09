import oci
import os
import json
import time
import random
from typing import Any, Dict, List
from app.services.oci_llm import call_oci_chat
from app.services.embedding_service import OCIEmbeddingService
from app.services.vector_store_service import search_similar_chunks
from app.services.secure_config import require_env, get_env


# ------ VM Configuration -------
# config = oci.config.from_file(
#     file_location="/home/opc/.oci/config",
#     profile_name="GC3TEST02"
# )


# ---- OCI Configuration ----
CONFIG_PROFILE = require_env("CONFIG_PROFILE")
OCI_CONFIG_PATH = get_env("OCI_CONFIG_PATH", os.path.expanduser("~/.oci/config"))
config = oci.config.from_file(
    file_location=OCI_CONFIG_PATH,
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
You are a data anonymization system. Your job is to redact sensitive info while preserving the original
format, structure, and wording as much as possible.

Rules:
- Replace all company, customer, client, partner, vendor, and organization names that are NOT "Oracle"
  (or obvious Oracle variants like "Oracle Cloud", "Oracle OCI") with [Anonymized Customer].
- Replace personal names, emails, phone numbers, account numbers, IDs, URLs, and IPs with [Anonymized].
- Do NOT change "Oracle" or its obvious variants.
- Do NOT add commentary, explanations, or extra text.
- Preserve punctuation, line breaks, markdown, tables, bullets, and headings.

Return ONLY the anonymized text.

Original Text:
{text}

Anonymized Text:
"""

    def _call_chat(message: str) -> str:
        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_request = oci.generative_ai_inference.models.CohereChatRequest()

        chat_request.message = message
        chat_request.max_tokens = 4000
        chat_request.temperature = 0

        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=MODEL_ID)
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = compartment_id

        response = generative_ai_inference_client.chat(chat_detail)
        return response.data.chat_response.text

    return _chat_with_retry(_call_chat, USER_MESSAGE)


def ai_redact_sensitive_info_batch(texts: List[str]) -> List[str]:
    """
    Batch anonymization. Returns a list of anonymized strings in the same order.
    """
    payload = json.dumps(texts, ensure_ascii=False)
    USER_MESSAGE = f"""
You are a data anonymization system. Anonymize each string in the JSON array below.

Rules:
- Replace all company, customer, client, partner, vendor, and organization names that are NOT "Oracle"
  (or obvious Oracle variants like "Oracle Cloud", "Oracle OCI") with [Anonymized Customer].
- Replace personal names, emails, phone numbers, account numbers, IDs, URLs, and IPs with [Anonymized].
- Do NOT change "Oracle" or its obvious variants.
- Do NOT add commentary, explanations, or extra text.
- Return ONLY a JSON array of strings, same length and order as the input.

Input JSON:
{payload}
"""

    def _call_chat(message: str) -> str:
        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_request = oci.generative_ai_inference.models.CohereChatRequest()

        chat_request.message = message
        chat_request.max_tokens = 4000
        chat_request.temperature = 0

        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=MODEL_ID)
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = compartment_id

        response = generative_ai_inference_client.chat(chat_detail)
        return response.data.chat_response.text

    raw = _chat_with_retry(_call_chat, USER_MESSAGE)
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _chat_with_retry(call_fn, message: str, max_retries: int = 5) -> str:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return call_fn(message)
        except Exception as e:
            last_err = e
            if not _is_rate_limit_error(e) or attempt == max_retries:
                raise
            _backoff_sleep(attempt)
    raise last_err


def _is_rate_limit_error(err: Exception) -> bool:
    if hasattr(err, "status") and err.status == 429:
        return True
    text = str(err)
    return "status': 429" in text or "code': '429" in text or "429" in text


def _backoff_sleep(attempt: int) -> None:
    base = min(60.0, (2 ** attempt))
    jitter = random.uniform(0, 0.5)
    time.sleep(base + jitter)

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
