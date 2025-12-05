import oci
import os
from typing import Any, Dict
from app.services.oci_llm import call_oci_chat
from app.services.embedding_service import OCIEmbeddingService
from app.services.vector_store_service import search_similar_chunks
from app.services.secure_config import require_env, get_env

# CONFIG_PROFILE = "GC3TEST02"
# CONFIG_PROFILE = require_env("CONFIG_PROFILE")
# config = oci.config.from_file(
#     file_location=r"C:\Users\shshrohi\.oci\config",
#     profile_name=CONFIG_PROFILE
# )

## for VM
config = oci.config.from_file(
    file_location="/home/opc/.oci/config",
    profile_name="GC3TEST02"
)


# compartment_id = "ocid1.compartment.oc1..aaaaaaaa2pf2tel6ftytyrdkwaareqpcjfyfit6s62v4qdukfjiflqhlmura"
compartment_id = require_env("COMPARTMENT_ID")
# MODEL_ID = "ocid1.generativeaimodel.oc1.ap-hyderabad-1.amaaaaaask7dceyaaccktjkitpfn3zp3xnkg6yclc6izeahggh2hkwawfjna"
MODEL_ID = require_env("MODEL_ID")
# endpoint = "https://inference.generativeai.ap-hyderabad-1.oci.oraclecloud.com"
endpoint = require_env("ENDPOINT")

generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config=config,
    service_endpoint=endpoint,
    retry_strategy=oci.retry.NoneRetryStrategy(),
    timeout=(10, 240)
)

def ai_redact_sensitive_info(text: str) -> str:
    """Send text to OCI Generative AI to anonymize sensitive info."""
    USER_MESSAGE = f"""
    You are a data anonymization expert. Replace all company/customer names
    that are NOT 'Oracle' with [Anonymized Customer].

    Return only the anonymized text.

    Original Text: {text}

    Anonymized Text:
    """
    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    chat_request = oci.generative_ai_inference.models.CohereChatRequest()
    chat_request.message = USER_MESSAGE
    chat_request.max_tokens = 4000
    chat_request.temperature = 1
    chat_request.frequency_penalty = 0
    chat_request.top_p = 0.75
    chat_request.top_k = 0

    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=MODEL_ID)
    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = compartment_id

    response = generative_ai_inference_client.chat(chat_detail)
    return response.data.chat_response.text

def answer_query(query: str, top_k: int = 5, llm_model: str = MODEL_ID) -> Dict[str, Any]:
    """
    End-to-end RAG: embed query -> vector search -> LLM answer
    """
    embedder = OCIEmbeddingService()

    query_embedding = embedder.embed_text(query)
    hits = search_similar_chunks(query_embedding, top_k=top_k)

    context = "\n\n".join([h["chunk"] for h in hits])

    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

CONTEXT:
{context}

QUESTION:
{query}

Answer concisely and reference which chunk the info came from when relevant.
"""

    llm_resp = call_oci_chat(prompt)
    answer_text = llm_resp
    return {"answer": answer_text, "chunks": hits}