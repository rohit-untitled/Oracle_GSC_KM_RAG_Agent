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


config_path = get_env("OCI_CONFIG_PATH", os.path.expanduser("~/.oci/config"))
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

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")


def _load_system_prompt() -> str:
    try:
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error("Failed to load system prompt from %s: %s", _PROMPT_PATH, e)
        return "You are a helpful assistant."


SYSTEM_PROMPT = _load_system_prompt()


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


def call_oci_title(message: str, max_words: int = 5) -> str:
    """
    Generate a short session title from a user message.
    """
    prompt = (
        "Generate a short, specific chat title (max "
        f"{max_words} words). Return only the title."
    )

    try:
        chat_request = CohereChatRequest(
            message=message,
            api_format=CohereChatRequest.API_FORMAT_COHERE,

            documents=[],
            chat_history=[
                {"role": "SYSTEM", "message": prompt},
            ],

            max_tokens=30,
            temperature=0.2,
            top_p=0.75,
            top_k=40,

            is_force_single_step=False,
            is_raw_prompting=False,
            is_search_queries_only=False,

            safety_mode=CohereChatRequest.SAFETY_MODE_CONTEXTUAL,
            prompt_truncation=CohereChatRequest.PROMPT_TRUNCATION_AUTO_PRESERVE_ORDER,
        )

        chat_details = ChatDetails(
            chat_request=chat_request,
            serving_mode=OnDemandServingMode(model_id=MODEL_ID),
            compartment_id=COMPARTMENT_ID,
        )

        response = oci_client.chat(chat_details)
        title = (response.data.chat_response.text or "").strip()
        return title

    except Exception as e:
        logger.error("OCI Title generation failed", exc_info=True)
        raise RuntimeError("LLM title generation failed") from e

