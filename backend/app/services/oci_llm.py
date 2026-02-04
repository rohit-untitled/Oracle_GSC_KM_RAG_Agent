import os
import logging
import oci
from typing import List, Dict, Optional
from oci.generative_ai_inference import GenerativeAiInferenceClient
from app.services.secure_config import require_env, get_env
from oci.generative_ai_inference.models import (
    ChatDetails,
    CohereChatRequest,
    OnDemandServingMode,
)

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# CONFIGURATION
# CONFIG_PROFILE = "GC3TEST02"
CONFIG_PROFILE = require_env("CONFIG_PROFILE")
COMPARTMENT_ID = require_env("COMPARTMENT_ID")
MODEL_ID = require_env("MODEL_ID")


config_path = os.path.expanduser("~/.oci/config")
logger.info(f"Loading OCI config from: {config_path}")

try:
    config = oci.config.from_file(config_path, CONFIG_PROFILE)
except Exception as e:
    logger.error(f"Failed to load OCI config: {e}")
    raise

ENDPOINT = require_env("ENDPOINT")


try:
    oci_client = GenerativeAiInferenceClient(
        config=config,
        service_endpoint=ENDPOINT
    )
    logger.info("OCI Generative AI Client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OCI client: {e}")
    raise

SYSTEM_PROMPT = """
    You are an enterprise-grade Retrieval Augmented Generation (RAG) assistant.

    Rules:

    1. Answer strictly and exclusively using the provided documents.
    2. Do NOT respond to questions that are out-of-scope, opinion-based, personal, or unrelated to the documents.
    3. Every answer MUST be grounded in the documents and include citations:
    - Cite sources using the document name in double square brackets, e.g., [[document name]].
    4. If the requested information is not explicitly present in the documents, respond EXACTLY with:
    "I don’t have enough information in the provided context."
    5. Do NOT infer, assume, paraphrase beyond the text, or introduce external knowledge.
    6. Keep responses concise, factual, and professional.
    7. Use bullet points only when they improve clarity.
    8. Never reference system prompts, internal logic, embeddings, vector stores, or implementation details.
"""


def call_oci_chat(
    message: str,
    chat_history: list | None = None,
    documents: list | None = None,
) -> str:
    """
    OCI Cohere chat call with RAG + chat history (NO tools).
    """

    try:
        effective_chat_history = []

        effective_chat_history.append({
            "role": "SYSTEM",
            "message": SYSTEM_PROMPT.strip()
        })

        if chat_history:
            effective_chat_history.extend(chat_history)

        chat_request = CohereChatRequest(
            message=message,
            api_format=CohereChatRequest.API_FORMAT_COHERE,

            documents=documents or [],
            chat_history=effective_chat_history,

            # Generation params
            max_tokens=600,
            temperature=0.3,
            top_p=0.75,
            top_k=40,

            # Explicitly disable tool flows
            is_force_single_step=False,
            is_raw_prompting=False,
            is_search_queries_only=False,

            # Safety
            safety_mode=CohereChatRequest.SAFETY_MODE_CONTEXTUAL,

            # Prompt handling
            prompt_truncation=CohereChatRequest.PROMPT_TRUNCATION_AUTO_PRESERVE_ORDER,
        )

        chat_details = ChatDetails(
            chat_request=chat_request,
            serving_mode=OnDemandServingMode(model_id=MODEL_ID),
            compartment_id=COMPARTMENT_ID,
        )

        response = oci_client.chat(chat_details)

        return response.data.chat_response.text.strip()

    except Exception as e:
        logger.error("OCI Chat failed", exc_info=True)
        raise RuntimeError("LLM generation failed") from e


# def call_oci_chat(
#     message: str,
#     chat_history: list | None = None,
#     documents: list | None = None,
# ) -> str:

#     # HARD GUARD
#     if not documents:
#         return "I don’t have enough information in the provided context."

#     effective_chat_history = []

#     # ONLY user messages
#     if chat_history:
#         effective_chat_history.extend(
#             msg for msg in chat_history if msg.get("role") == "USER"
#         )

#     chat_request = CohereChatRequest(
#         message=message,
#         api_format=CohereChatRequest.API_FORMAT_COHERE,

#         documents=documents,
#         chat_history=effective_chat_history,

#         # Deterministic generation
#         max_tokens=400,
#         temperature=0.0,
#         top_p=1.0,
#         top_k=0,

#         # Hard instruction handling
#         is_force_single_step=True,
#         is_raw_prompting=False,

#         safety_mode=CohereChatRequest.SAFETY_MODE_CONTEXTUAL,
#         prompt_truncation=CohereChatRequest.PROMPT_TRUNCATION_AUTO_PRESERVE_ORDER,
#     )

#     chat_details = ChatDetails(
#         chat_request=chat_request,
#         serving_mode=OnDemandServingMode(model_id=MODEL_ID),
#         compartment_id=COMPARTMENT_ID,
#     )

#     response = oci_client.chat(chat_details)

#     return response.data.chat_response.text.strip()