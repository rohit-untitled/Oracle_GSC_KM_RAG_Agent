import logging
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.oci_llm import LLMTimeoutError
from app.services.rag_service import answer_query
from app.services.request_context import get_request_id, reset_request_id, set_request_id
from app.services.secure_config import get_env


router = APIRouter()
logger = logging.getLogger("KM Knowledge Agent is Working")

ASK_MAX_TURNS = int(get_env("ASK_MAX_TURNS", "15"))
ASK_MAX_INPUT_TOKENS = int(get_env("ASK_MAX_INPUT_TOKENS", "4000"))


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RAGRequest(BaseModel):
    query: str
    top_k: Optional[int] = Field(None, ge=1, le=50)
    session_id: Optional[str] = None
    history: List[ChatTurn] = Field(default_factory=list)
    generate_title: bool = False
    model: Literal["cohere", "maverick", "gpt-5.2"] = "maverick"
    mode: Literal["instant", "thinking", "pro"] = "instant"
    confidentiality: Literal["SCM", "ERP", "EPM"] = "SCM"


class ChatModelOption(BaseModel):
    key: Literal["cohere", "maverick", "gpt-5.2"]
    label: str
    is_default: bool = False


class ChatModelsResponse(BaseModel):
    models: List[ChatModelOption]


def _estimate_tokens(text: str) -> int:
    # Lightweight approximation for server-side budget control.
    return max(1, len((text or "").strip()) // 4)


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _log_ask_timing(stage: str, started_at: float, **fields: Any) -> None:
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    if field_text:
        logger.info(
            "ask_timing | request_id=%s | stage=%s | elapsed_ms=%s | %s",
            get_request_id(),
            stage,
            _elapsed_ms(started_at),
            field_text,
        )
    else:
        logger.info(
            "ask_timing | request_id=%s | stage=%s | elapsed_ms=%s",
            get_request_id(),
            stage,
            _elapsed_ms(started_at),
        )


def _normalize_history(history: List[ChatTurn]) -> List[Dict[str, str]]:
    return [{"role": turn.role, "content": turn.content} for turn in history]


def _normalize_turn(turn: Dict[str, str]) -> Optional[Dict[str, str]]:
    role = (turn.get("role") or "").strip().lower()
    content = (turn.get("content") or "").strip()
    if role not in {"user", "assistant"} or not content:
        return None
    return {"role": role, "content": content}


def _sanitize_request_history(request_history: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    normalized_history = [
        normalized
        for turn in request_history
        if (normalized := _normalize_turn(turn)) is not None
    ]

    current_query = (query or "").strip()
    if normalized_history and normalized_history[-1]["role"] == "user":
        if normalized_history[-1]["content"].strip() == current_query:
            normalized_history = normalized_history[:-1]

    deduped: List[Dict[str, str]] = []
    for turn in normalized_history:
        if deduped and deduped[-1] == turn:
            continue
        deduped.append(turn)
    return deduped


def _apply_history_limits(
    history: List[Dict[str, str]],
    query: str,
    max_turns: int,
    max_input_tokens: int,
) -> tuple[List[Dict[str, str]], int, int]:
    limited_history = history[-max_turns:] if max_turns > 0 else history
    query_tokens = _estimate_tokens(query)
    if query_tokens >= max_input_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Query is too long for token budget ({max_input_tokens}).",
        )

    budget_for_history = max_input_tokens - query_tokens
    history_tokens = sum(_estimate_tokens(turn.get("content", "")) for turn in limited_history)

    # Drop oldest turns until budget fits.
    while limited_history and history_tokens > budget_for_history:
        removed = limited_history.pop(0)
        history_tokens -= _estimate_tokens(removed.get("content", ""))

    return limited_history, history_tokens, query_tokens


def _process_ask(
    payload: RAGRequest,
    query: str,
    history_payload: List[Dict[str, str]],
    history_tokens: int,
    query_tokens: int,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    logger.info(
        "ask_timing | request_id=%s | stage=ask.answer_query_start | mode=%s model=%s history_turns=%s query_tokens=%s history_tokens=%s",
        get_request_id(),
        payload.mode,
        payload.model,
        len(history_payload),
        query_tokens,
        history_tokens,
    )
    response = answer_query(
        query=query,
        top_k=payload.top_k,
        history=history_payload,
        model=payload.model,
        generate_title=payload.generate_title,
        mode=payload.mode,
        confidentiality=payload.confidentiality,
    )
    elapsed_seconds = time.perf_counter() - started_at
    logger.info(
        "ask_timing | request_id=%s | stage=ask.answer_query_complete | elapsed_ms=%s | mode=%s model=%s",
        get_request_id(),
        int(elapsed_seconds * 1000),
        response.get("retrieval_config", {}).get("mode", payload.mode),
        response.get("model_used", payload.model),
    )
    return {
        "session_id": payload.session_id,
        "request_id": get_request_id(),
        "answer": response["answer"],
        "chunks": response["chunks"],
        "citations": response.get("citations", []),
        "history_length": response["history_length"],
        "model": response["model_used"],
        "generated_title": response.get("generated_title"),
        "mode": response.get("retrieval_config", {}).get("mode", payload.mode),
        "confidentiality": response.get("confidentiality", payload.confidentiality),
        "time_taken": round(elapsed_seconds, 3),
        "token_usage": {
            "input_tokens_estimated": history_tokens + query_tokens,
            "history_tokens_estimated": history_tokens,
            "query_tokens_estimated": query_tokens,
            "max_input_tokens": ASK_MAX_INPUT_TOKENS,
        },
    }


@router.get("/chat-models", response_model=ChatModelsResponse)
def list_chat_models():
    return {
        "models": [
            {"key": "cohere", "label": "Cohere (Default)", "is_default": True},
            {"key": "maverick", "label": "Maverick", "is_default": False},
            {"key": "gpt-5.2", "label": "GPT-5.2", "is_default": False},
        ]
    }


@router.post("/ask")
def ask_endpoint(payload: RAGRequest):
    request_started_at = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    request_token = set_request_id(request_id)
    query = (payload.query or "").strip()
    if not query:
        reset_request_id(request_token)
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    logger.info(
        "Received ask request | request_id=%s model=%s confidentiality=%s mode=%s session_id=%s query_chars=%s history_turns=%s",
        get_request_id(),
        payload.model,
        payload.confidentiality,
        payload.mode,
        payload.session_id or "-",
        len(query),
        len(payload.history or []),
    )

    try:
        prep_started_at = time.perf_counter()
        history_payload = _normalize_history(payload.history)
        history_payload = _sanitize_request_history(history_payload, query)
        history_payload, history_tokens, query_tokens = _apply_history_limits(
            history=history_payload,
            query=query,
            max_turns=ASK_MAX_TURNS,
            max_input_tokens=ASK_MAX_INPUT_TOKENS,
        )
        _log_ask_timing(
            "ask.prepare_request",
            prep_started_at,
            history_turns=len(history_payload),
            query_tokens=query_tokens,
            history_tokens=history_tokens,
        )
        result = _process_ask(
            payload=payload,
            query=query,
            history_payload=history_payload,
            history_tokens=history_tokens,
            query_tokens=query_tokens,
        )
        _log_ask_timing(
            "ask.total",
            request_started_at,
            mode=result.get("mode"),
            model=result.get("model"),
            time_taken=result.get("time_taken"),
        )
        return result
    except HTTPException:
        raise
    except LLMTimeoutError as e:
        logger.warning("LLM timeout while processing ask request | request_id=%s error=%s", get_request_id(), e)
        raise HTTPException(
            status_code=504,
            detail={
                "message": "The knowledge assistant took too long to respond.",
                "error": "LLM request timed out. Please retry, use a shorter question, or try a different model.",
                "request_id": get_request_id(),
            },
        )
    except Exception as e:
        logger.exception("Error in RAG query | request_id=%s error=%s", get_request_id(), e)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to process ask request.",
                "error": "Internal server error while generating response.",
                "request_id": get_request_id(),
            },
        )
    finally:
        reset_request_id(request_token)
