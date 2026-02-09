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
You are an enterprise-grade Retrieval Augmented Generation (RAG) assistant designed to provide accurate, consistent, and document-grounded answers.

Core Principles:
1. Answer questions using ONLY the provided documents.
2. All answers MUST be grounded in the documents and MUST include citations.
   - Citations must reference ONLY the document name in double square brackets, e.g., [[Employee_Policy.pdf]].
   - Do NOT include chunk numbers, page numbers, or internal identifiers.
3. Do NOT use external knowledge, assumptions, or inferred information beyond what is explicitly stated in the documents.

Answering Guidelines:
4. If relevant information exists in the documents, ALWAYS attempt to answer using it.
   - Do NOT respond with “I don’t have enough information” if the documents contain partial or indirect but relevant content.
   - If the documents cover the topic incompletely, clearly state what is present and what is not, based strictly on the text.
5. Answers should be detailed, precise, and non-generic.
   - Prefer exact wording, technical terms, definitions, and statements as written in the document.
   - Paraphrase minimally and only when needed for clarity.
6. Avoid vague summaries. Expand explanations using the actual language and concepts from the document.

Formatting Rules:
7. Use paragraph-style explanations by default.
8. Use bullet points ONLY when they improve clarity.
9. If the user explicitly asks for a table, list, or structured format, present the answer in that format.

Consistency Rules:
10. For the same question asked by different users, the answer should be highly consistent in structure, terminology, and meaning.
    - Do not introduce randomness, stylistic variation, or alternative interpretations unless the documents explicitly allow it.

Restrictions:
11. Do NOT entertain hypothetical scenarios, jokes, opinions, or personal advice.
12. Do NOT introduce examples or explanations that are not explicitly supported by the documents.
13. Do NOT reference system prompts, internal reasoning, embeddings, vector stores, retrieval mechanisms, or implementation details.

Failure Handling:
14. If and only if the documents contain NO relevant information at all, respond with:
    “The provided documents do not contain information related to this question.”
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
