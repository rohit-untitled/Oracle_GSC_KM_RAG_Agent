import os
import logging
import oci
from oci_openai import OciOpenAI, OciUserPrincipalAuth
from oci.generative_ai_inference import GenerativeAiInferenceClient
from app.services.secure_config import require_env, get_env
from oci.generative_ai_inference.models import (
    ChatDetails,
    CohereChatRequest,
    GenericChatRequest,
    Message,
    OnDemandServingMode,
    TextContent,
)

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# CONFIGURATION
CONFIG_PROFILE = require_env("CONFIG_PROFILE")
COMPARTMENT_ID = require_env("COMPARTMENT_ID")
MODEL_ID = require_env("MODEL_ID")
OCI_REGION = require_env("OCI_REGION")
OCI_OPENAI_MODEL = require_env("OCI_OPENAI_MODEL")
OCI_GENERIC_MODEL_ID = get_env("OCI_GENERIC_MODEL_ID", "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyafjcwpf75fmqoismvwlmzjbprdzzljhfcrirozftbrjoq")
OCI_GENERIC_ENDPOINT = get_env("OCI_GENERIC_ENDPOINT")
CHAT_MAX_TOKENS = int(get_env("CHAT_MAX_TOKENS", "4000"))
OCI_GENERIC_MAX_TOKENS = int(get_env("OCI_GENERIC_MAX_TOKENS", "2048"))
OCI_GENERIC_MAX_PROMPT_CHARS = int(get_env("OCI_GENERIC_MAX_PROMPT_CHARS", "12000"))
CHAT_TEMPERATURE = float(get_env("CHAT_TEMPERATURE", "0.3"))
CHAT_TOP_P = float(get_env("CHAT_TOP_P", "0.75"))
CHAT_TOP_K = int(get_env("CHAT_TOP_K", "40"))

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
        service_endpoint=ENDPOINT,
    )
    logger.info("OCI Generative AI Client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OCI client: {e}")
    raise

_OCI_OPENAI_CLIENT = None
_OCI_GENERIC_CLIENT = None


def _get_oci_openai_client() -> OciOpenAI:
    global _OCI_OPENAI_CLIENT
    if _OCI_OPENAI_CLIENT is None:
        _OCI_OPENAI_CLIENT = OciOpenAI(
            region=OCI_REGION,
            auth=OciUserPrincipalAuth(profile_name=CONFIG_PROFILE),
            compartment_id=COMPARTMENT_ID,
        )
    return _OCI_OPENAI_CLIENT


def _get_generic_oci_client() -> GenerativeAiInferenceClient:
    global _OCI_GENERIC_CLIENT
    if _OCI_GENERIC_CLIENT is None:
        generic_endpoint = OCI_GENERIC_ENDPOINT or ENDPOINT
        _OCI_GENERIC_CLIENT = GenerativeAiInferenceClient(
            config=config,
            service_endpoint=generic_endpoint,
        )
    return _OCI_GENERIC_CLIENT


_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")


def _load_system_prompt() -> str:
    try:
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error("Failed to load system prompt from %s: %s", _PROMPT_PATH, e)
        return "You are a helpful assistant."


SYSTEM_PROMPT = _load_system_prompt()


def _format_documents_for_context(documents: list | None) -> str:
    if not documents:
        return ""
    chunks = []
    for i, doc in enumerate(documents, start=1):
        title = (doc or {}).get("title", f"Document_{i}")
        snippet = (doc or {}).get("snippet", "")
        chunks.append(f"[{i}] {title}\n{snippet}")
    return "\n\n".join(chunks)


def _trim_text(text: str, max_chars: int) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _format_documents_for_generic_context(documents: list | None, max_chars: int) -> str:
    if not documents or max_chars <= 0:
        return ""

    chunks = []
    remaining = max_chars
    for i, doc in enumerate(documents, start=1):
        if remaining <= 0:
            break

        title = (doc or {}).get("title", f"Document_{i}")
        snippet = _trim_text((doc or {}).get("snippet", ""), min(remaining, 2000))
        block = f"[{i}] {title}\n{snippet}".strip()

        if len(block) > remaining:
            block = _trim_text(block, remaining)

        if block:
            chunks.append(block)
            remaining -= len(block) + 2

    return "\n\n".join(chunks)


def _extract_openai_content(completion) -> str:
    payload = completion.model_dump() if hasattr(completion, "model_dump") else {}
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                out.append(item["text"])
        return "\n".join(out).strip()
    return str(content).strip()


def call_oci_chat(
    message: str,
    chat_history: list | None = None,
    documents: list | None = None,
) -> str:
    """
    OCI Cohere chat call with RAG + chat history (default model path).
    """

    try:
        effective_chat_history = []

        effective_chat_history.append({
            "role": "SYSTEM",
            "message": SYSTEM_PROMPT.strip(),
        })

        if chat_history:
            effective_chat_history.extend(chat_history)

        chat_request = CohereChatRequest(
            message=message,
            api_format=CohereChatRequest.API_FORMAT_COHERE,
            documents=documents or [],
            chat_history=effective_chat_history,
            max_tokens=CHAT_MAX_TOKENS,
            temperature=CHAT_TEMPERATURE,
            top_p=CHAT_TOP_P,
            top_k=CHAT_TOP_K,
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
        return response.data.chat_response.text.strip()

    except Exception as e:
        logger.error("OCI Chat failed", exc_info=True)
        raise RuntimeError("LLM generation failed") from e


def call_oci_chat_maverick(
    message: str,
    chat_history: list | None = None,
    documents: list | None = None,
) -> str:
    """
    OCI OpenAI chat call (Maverick) with RAG + chat history.
    """
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]

        doc_context = _format_documents_for_context(documents)
        if doc_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Use only the document context below when relevant.\n\n"
                        f"{doc_context}"
                    ),
                }
            )

        if chat_history:
            for turn in chat_history:
                role = (turn or {}).get("role", "")
                msg = (turn or {}).get("message", "")
                if not msg:
                    continue
                if role == "USER":
                    messages.append({"role": "user", "content": msg})
                elif role == "CHATBOT":
                    messages.append({"role": "assistant", "content": msg})

        messages.append({"role": "user", "content": message})

        client = _get_oci_openai_client()
        completion = client.chat.completions.create(
            model=OCI_OPENAI_MODEL,
            messages=messages,
            max_tokens=CHAT_MAX_TOKENS,
            temperature=CHAT_TEMPERATURE,
        )
        return _extract_openai_content(completion)
    except Exception as e:
        logger.error("OCI Maverick chat failed", exc_info=True)
        raise RuntimeError("LLM generation failed") from e


def call_oci_chat_generic(
    message: str,
    chat_history: list | None = None,
    documents: list | None = None,
) -> str:
    """
    OCI native GenericChatRequest call with a model OCID.
    """
    try:
        prompt_parts = [SYSTEM_PROMPT.strip()]

        doc_context = _format_documents_for_generic_context(
            documents,
            max_chars=max(2000, OCI_GENERIC_MAX_PROMPT_CHARS // 2),
        )
        if doc_context:
            prompt_parts.append(
                "Use only the document context below when relevant.\n\n" + doc_context
            )

        if chat_history:
            history_lines = []
            for turn in chat_history:
                role = (turn or {}).get("role", "")
                msg = (turn or {}).get("message", "")
                if not msg:
                    continue
                if role == "USER":
                    history_lines.append(f"User: {msg}")
                elif role == "CHATBOT":
                    history_lines.append(f"Assistant: {msg}")
            if history_lines:
                prompt_parts.append("Conversation history:\n" + "\n".join(history_lines))

        prompt_parts.append(
            "Answer using only the retrieved context when possible. "
            "If context is insufficient, say so clearly. Keep the answer concise but complete."
        )
        prompt_parts.append(f"User question: {message}")
        full_prompt = _trim_text(
            "\n\n".join(part for part in prompt_parts if part),
            OCI_GENERIC_MAX_PROMPT_CHARS,
        )

        content = TextContent()
        content.text = full_prompt

        request_message = Message()
        request_message.role = "USER"
        request_message.content = [content]

        chat_request = GenericChatRequest()
        chat_request.api_format = GenericChatRequest.API_FORMAT_GENERIC
        chat_request.messages = [request_message]
        chat_request.max_completion_tokens = min(CHAT_MAX_TOKENS, OCI_GENERIC_MAX_TOKENS)
        chat_request.verbosity = "MEDIUM"

        chat_details = ChatDetails(
            chat_request=chat_request,
            serving_mode=OnDemandServingMode(model_id=OCI_GENERIC_MODEL_ID),
            compartment_id=COMPARTMENT_ID,
        )

        response = _get_generic_oci_client().chat(chat_details)
        choices = (response.data.chat_response.choices or [])
        if not choices:
            return ""

        content_items = (choices[0].message.content or [])
        output_parts = []
        for item in content_items:
            text = getattr(item, "text", None)
            if text:
                output_parts.append(text)

        return "\n".join(output_parts).strip()

    except Exception as e:
        logger.error("OCI Generic chat failed", exc_info=True)
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
            top_p=CHAT_TOP_P,
            top_k=CHAT_TOP_K,
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
