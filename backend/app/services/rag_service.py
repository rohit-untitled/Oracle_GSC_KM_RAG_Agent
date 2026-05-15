import logging
import re
from typing import Any, Dict, List, Optional

from app.services.embedding_service import OCIEmbeddingService
from app.services.oci_llm import (
    call_oci_chat,
    call_oci_chat_generic,
    call_oci_chat_maverick,
    call_oci_title,
)
from app.services.retrieval_service import search_similar_chunks
from app.services.secure_config import get_env


logger = logging.getLogger(__name__)

HISTORY_TURNS = int(get_env("HISTORY_TURNS", "15"))
DEFAULT_CHAT_MODEL = "cohere"
SUPPORTED_CHAT_MODELS = {"cohere", "maverick", "gpt-5.2"}
DEFAULT_TOP_K = int(get_env("DEFAULT_TOP_K", "8"))
SUPPORTED_CONFIDENTIALITY = {"SCM", "ERP", "EPM"}
MAX_HISTORY_TURN_CHARS = int(get_env("MAX_HISTORY_TURN_CHARS", "1500"))
MAX_RETRIEVAL_QUERY_CHARS = int(get_env("MAX_RETRIEVAL_QUERY_CHARS", "500"))
MAX_RETRIEVAL_CONTEXT_TURNS = int(get_env("MAX_RETRIEVAL_CONTEXT_TURNS", "2"))
MIN_RETRIEVAL_TOKEN_OVERLAP = int(get_env("MIN_RETRIEVAL_TOKEN_OVERLAP", "1"))

FOLLOWUP_PATTERNS = [
    r"\bit\b",
    r"\bthis\b",
    r"\bthat\b",
    r"\bthese\b",
    r"\bthose\b",
    r"\bthey\b",
    r"\bthem\b",
    r"\bboth\b",
    r"\bformer\b",
    r"\blatter\b",
    r"\bprevious\b",
    r"\babove\b",
    r"\bearlier\b",
    r"\bsame\b",
    r"\bagain\b",
    r"\bmore\b",
    r"\bfurther\b",
    r"\belaborate\b",
    r"\bexpand\b",
    r"\bsummarize\b",
    r"\bexplain\s+more\b",
    r"\btell\s+me\s+more\b",
    r"\bwhat\s+about\b",
    r"\bwhich\s+one\b",
    r"\bcompare\b",
    r"\blike\s+a\s+flow\b",
    r"\bin\s+simple\s+terms\b",
]

DOMAIN_HINT_PATTERNS = [
    r"\bnpi\b",
    r"\boda\b",
    r"\bai\s+agent\b",
    r"\boracle\b",
    r"\bfusion\b",
    r"\berp\b",
    r"\bscm\b",
    r"\bepm\b",
    r"\bintegration\b",
    r"\bworkflow\b",
    r"\bprocess\b",
    r"\bapproval\b",
    r"\bforecast\b",
    r"\bplanning\b",
    r"\bprocurement\b",
    r"\bsupply\b",
    r"\bdocument\b",
    r"\bworkbook\b",
    r"\bmapping\b",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "how", "i",
    "in", "is", "it", "like", "me", "of", "on", "or", "please", "show", "steps", "tell",
    "that", "the", "this", "to", "us", "what", "which", "with", "you", "your",
}

SOURCE_QUESTION_PATTERNS = [
    r"\bwhat\s+is\s+the\s+source\b",
    r"\bwhat\s+is\s+the\s+source\s+of\s+this\b",
    r"\bsource\s+of\s+this\b",
    r"\bwhich\s+document\b",
    r"\bwhere\s+did\s+you\s+get\s+this\b",
    r"\bwhere\s+is\s+this\s+from\b",
    r"\bshow\s+the\s+source\b",
    r"\bcite\s+the\s+source\b",
]

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


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _truncate_text(value: str, max_chars: int) -> str:
    normalized = _normalize_spaces(value)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _clean_history_turns(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for turn in history:
        role = ((turn or {}).get("role") or "").strip().lower()
        content = _truncate_text((turn or {}).get("content") or "", MAX_HISTORY_TURN_CHARS)
        if role not in {"user", "assistant"} or not content:
            continue
        if cleaned and cleaned[-1]["role"] == role and cleaned[-1]["content"] == content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _tokenize_keywords(text: str) -> List[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9_\-/]+", (text or "").lower()):
        if len(token) <= 2 or token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _has_domain_hints(query: str) -> bool:
    q = (query or "").lower()
    return any(re.search(pattern, q) for pattern in DOMAIN_HINT_PATTERNS)


def _is_source_question(query: str) -> bool:
    q = _normalize_spaces(query).lower()
    return any(re.search(pattern, q) for pattern in SOURCE_QUESTION_PATTERNS)


def _detect_followup_signals(query: str) -> bool:
    q = _normalize_spaces(query).lower()
    if not q:
        return False
    if len(q.split()) <= 3 and not _has_domain_hints(q):
        return True
    return any(re.search(pattern, q) for pattern in FOLLOWUP_PATTERNS)


def _extract_recent_user_turns(
    history: List[Dict[str, str]],
    max_turns: int = MAX_RETRIEVAL_CONTEXT_TURNS,
) -> List[str]:
    recent: List[str] = []
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        content = _normalize_spaces(turn.get("content", ""))
        if len(content) < 4:
            continue
        recent.append(content)
        if len(recent) >= max(1, max_turns):
            break
    recent.reverse()
    return recent


def _query_is_standalone(query: str) -> bool:
    q = _normalize_spaces(query)
    if not q:
        return False
    if q.endswith("?") and len(q.split()) >= 5:
        return True
    if _has_domain_hints(q):
        return True
    keyword_count = len(_tokenize_keywords(q))
    return keyword_count >= 4 and not _detect_followup_signals(q)


def _detect_topic_shift(query: str, recent_user_turns: List[str]) -> bool:
    if not recent_user_turns:
        return False
    if not _query_is_standalone(query):
        return False
    current_tokens = set(_tokenize_keywords(query))
    previous_tokens = set(_tokenize_keywords(" ".join(recent_user_turns)))
    if not current_tokens:
        return False
    return current_tokens.isdisjoint(previous_tokens) and _has_domain_hints(query)


def _classify_query_context(query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    normalized_query = _normalize_spaces(query)
    recent_user_turns = _extract_recent_user_turns(history)
    source_question = _is_source_question(normalized_query)
    followup = _detect_followup_signals(normalized_query)
    standalone = _query_is_standalone(normalized_query)
    topic_shift = _detect_topic_shift(normalized_query, recent_user_turns)

    if source_question and recent_user_turns:
        classification = "source_question"
        confidence = 0.95
        reason = "query explicitly asks for the source of prior information"
    elif topic_shift:
        classification = "topic_shift"
        confidence = 0.9
        reason = "current query is self-contained and does not overlap with recent user topics"
    elif followup and recent_user_turns:
        classification = "followup"
        confidence = 0.85
        reason = "query contains follow-up signals and recent user topic exists"
    elif standalone:
        classification = "standalone"
        confidence = 0.8
        reason = "query contains enough standalone topic signals"
    elif recent_user_turns:
        classification = "ambiguous"
        confidence = 0.5
        reason = "query may depend on prior turns but confidence is limited"
    else:
        classification = "standalone"
        confidence = 0.6
        reason = "no usable history available"

    return {
        "classification": classification,
        "confidence": confidence,
        "reason": reason,
        "recent_user_turns": recent_user_turns,
    }


def _rewrite_followup_query(query: str, recent_user_turns: List[str]) -> str:
    current = _normalize_spaces(query)
    if not recent_user_turns:
        return current

    recent_topic = recent_user_turns[-1]
    lowered = current.lower()

    if "like a flow" in lowered and "steps" in lowered and "process" in recent_topic.lower():
        return _truncate_text(f"show me the steps in {recent_topic} like a flow", MAX_RETRIEVAL_QUERY_CHARS)

    if re.search(r"\bwhich\s+one\b|\bboth\b|\bcompare\b", lowered):
        return _truncate_text(f"{current} in the context of {recent_topic}", MAX_RETRIEVAL_QUERY_CHARS)

    if re.search(r"\bthis\b|\bthat\b|\bit\b|\bthem\b|\bthose\b|\bthese\b", lowered):
        return _truncate_text(f"{current} about {recent_topic}", MAX_RETRIEVAL_QUERY_CHARS)

    return _truncate_text(f"{recent_topic} {current}", MAX_RETRIEVAL_QUERY_CHARS)


def _build_retrieval_candidates(query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    normalized_query = _truncate_text(query, MAX_RETRIEVAL_QUERY_CHARS)
    context = _classify_query_context(normalized_query, history)
    classification = context["classification"]
    recent_user_turns = context["recent_user_turns"]

    candidates: List[str] = [normalized_query]

    if classification == "source_question" and recent_user_turns:
        for topic in reversed(recent_user_turns[-2:]):
            topical = _truncate_text(f"source document for {topic}", MAX_RETRIEVAL_QUERY_CHARS)
            if topical and topical not in candidates:
                candidates.insert(0, topical)
    elif classification == "standalone":
        candidates = [normalized_query]
    elif classification == "topic_shift":
        candidates = [normalized_query]
    elif classification == "followup":
        rewritten = _rewrite_followup_query(normalized_query, recent_user_turns)
        if rewritten and rewritten not in candidates:
            candidates.insert(0, rewritten)
    elif classification == "ambiguous" and recent_user_turns:
        conservative = _rewrite_followup_query(normalized_query, [recent_user_turns[-1]])
        if conservative and conservative not in candidates:
            candidates.append(conservative)

    if classification in {"followup", "ambiguous", "source_question"}:
        for topic in reversed(recent_user_turns[-1:]):
            topical = _truncate_text(topic, MAX_RETRIEVAL_QUERY_CHARS)
            if topical and topical not in candidates:
                candidates.append(topical)

    return {
        "original_query": normalized_query,
        "candidates": candidates[:3],
        "classification": classification,
        "confidence": context["confidence"],
        "reason": context["reason"],
    }


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


def _format_citation_label(hit: Dict[str, Any]) -> Optional[str]:
    metadata = (hit or {}).get("metadata", {}) or {}
    source_file = metadata.get("source_file")
    heading = metadata.get("heading")
    sheet_name = metadata.get("sheet_name")
    row_number = metadata.get("row_number")
    row_start = metadata.get("row_start")
    row_end = metadata.get("row_end")

    if sheet_name and row_start is not None and row_end is not None:
        return f"[Workbook: {source_file or 'Unknown'} | Sheet: {sheet_name} | Rows: {row_start}-{row_end}]"
    if sheet_name and row_number is not None:
        return f"[Workbook: {source_file or 'Unknown'} | Sheet: {sheet_name} | Row: {row_number}]"
    if heading:
        return f"[Doc: {source_file or 'Unknown'} | Section: {heading}]"
    if source_file:
        return f"[Doc: {source_file}]"
    return None


def _answer_has_citation(answer: str) -> bool:
    text = answer or ""
    return any(marker in text for marker in ["[Doc:", "[Workbook:", "[Source:"])


def _should_expect_citation(answer: str, hits: List[Dict[str, Any]]) -> bool:
    if not answer or not hits:
        return False
    lowered = answer.lower()
    no_context_markers = [
        "i couldn’t find matching project material",
        "i couldn't find matching project material",
        "no matching knowledge found",
    ]
    return not any(marker in lowered for marker in no_context_markers)


def _enforce_citations(answer: str, hits: List[Dict[str, Any]]) -> str:
    if not _should_expect_citation(answer, hits):
        return answer
    if _answer_has_citation(answer):
        return answer

    citation_labels: List[str] = []
    for hit in hits[:3]:
        label = _format_citation_label(hit)
        if label and label not in citation_labels:
            citation_labels.append(label)

    if not citation_labels:
        return answer

    suffix = "\n\nSources: " + ", ".join(citation_labels)
    return (answer or "").rstrip() + suffix


def _resolve_retrieval_profile(mode: str | None, top_k_override: int | None) -> Dict[str, Any]:
    mode_key = (mode or "thinking").strip().lower()
    if mode_key not in RETRIEVAL_PROFILES:
        mode_key = "thinking"

    profile = dict(RETRIEVAL_PROFILES[mode_key])
    if top_k_override is not None:
        profile["top_k"] = top_k_override
    profile["mode"] = mode_key
    return profile


def _run_retrieval(
    embedder: OCIEmbeddingService,
    retrieval_query: str,
    retrieval_profile: Dict[str, Any],
    confidentiality_key: str,
) -> List[Dict[str, Any]]:
    query_embedding = embedder.embed_text(retrieval_query)
    if not query_embedding:
        return []
    return search_similar_chunks(
        query_embedding,
        top_k=retrieval_profile["top_k"],
        query_text=retrieval_query,
        rerank_top_n=retrieval_profile["rerank_top_n"],
        neighbor_radius=retrieval_profile["neighbor_radius"],
        use_hybrid=retrieval_profile["use_hybrid"],
        confidentiality=confidentiality_key,
    )


def _score_hit_relevance(hit: Dict[str, Any], query: str) -> int:
    query_tokens = set(_tokenize_keywords(query))
    if not query_tokens:
        return 0
    chunk_text = _normalize_spaces((hit or {}).get("chunk", ""))
    metadata = (hit or {}).get("metadata", {}) or {}
    source_file = _normalize_spaces(metadata.get("source_file", ""))
    heading = _normalize_spaces(metadata.get("heading", ""))
    combined = " ".join([chunk_text, source_file, heading]).lower()
    return sum(1 for token in query_tokens if token in combined)


def _retrieval_has_sufficient_signal(hits: List[Dict[str, Any]], query: str) -> bool:
    if not hits:
        return False
    best_score = max(_score_hit_relevance(hit, query) for hit in hits)
    return best_score >= MIN_RETRIEVAL_TOKEN_OVERLAP


def _merge_hits(hit_groups: List[List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for hits in hit_groups:
        for hit in hits:
            metadata = (hit or {}).get("metadata", {}) or {}
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                continue
            if chunk_id not in merged:
                merged[chunk_id] = hit
                order.append(chunk_id)
    return [merged[chunk_id] for chunk_id in order[:top_k]]


def _retrieve_with_history_awareness(
    *,
    query: str,
    incoming_history: List[Dict[str, str]],
    retrieval_profile: Dict[str, Any],
    confidentiality_key: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    embedder = OCIEmbeddingService()
    plan = _build_retrieval_candidates(query, incoming_history)
    hit_groups: List[List[Dict[str, Any]]] = []
    attempted_queries: List[str] = []

    winning_query: Optional[str] = None
    winning_hits: List[Dict[str, Any]] = []
    for candidate in plan["candidates"]:
        attempted_queries.append(candidate)
        hits = _run_retrieval(embedder, candidate, retrieval_profile, confidentiality_key)
        hit_groups.append(hits)
        if hits and _retrieval_has_sufficient_signal(hits, candidate):
            winning_query = candidate
            winning_hits = hits
            break

    merged_hits = _merge_hits(hit_groups, retrieval_profile["top_k"])
    if winning_hits:
        merged_hits = _merge_hits([winning_hits], retrieval_profile["top_k"])

    if not merged_hits and retrieval_profile["mode"] == "instant":
        fallback_profile = dict(retrieval_profile)
        fallback_profile.update({
            "top_k": 6,
            "rerank_top_n": 10,
            "neighbor_radius": 1,
            "use_hybrid": True,
        })
        fallback_groups: List[List[Dict[str, Any]]] = []
        for candidate in attempted_queries[:2]:
            fallback_hits = _run_retrieval(embedder, candidate, fallback_profile, confidentiality_key)
            fallback_groups.append(fallback_hits)
            if fallback_hits and _retrieval_has_sufficient_signal(fallback_hits, candidate):
                winning_query = candidate
                winning_hits = fallback_hits
                break
        merged_hits = _merge_hits(fallback_groups, fallback_profile["top_k"])
        if winning_hits:
            merged_hits = _merge_hits([winning_hits], fallback_profile["top_k"])

    retrieval_debug = {
        "original_query": plan["original_query"],
        "candidate_queries": attempted_queries,
        "classification": plan["classification"],
        "classification_confidence": plan["confidence"],
        "classification_reason": plan["reason"],
        "hit_count": len(merged_hits),
        "winning_query": winning_query,
        "relevance_passed": bool(merged_hits),
        "top_sources": [
            ((hit or {}).get("metadata", {}) or {}).get("source_file")
            for hit in merged_hits[:3]
        ],
    }
    logger.info(
        "Retrieval plan | classification=%s | original=%s | candidates=%s | winning_query=%s | hits=%s | sources=%s",
        retrieval_debug["classification"],
        retrieval_debug["original_query"],
        retrieval_debug["candidate_queries"],
        retrieval_debug["winning_query"],
        retrieval_debug["hit_count"],
        retrieval_debug["top_sources"],
    )
    return merged_hits, retrieval_debug


def _select_generation_history(
    incoming_history: List[Dict[str, str]],
    retrieval_debug: Dict[str, Any],
) -> List[Dict[str, str]]:
    classification = retrieval_debug.get("classification")
    if classification in {"standalone", "topic_shift"}:
        return []
    if classification == "source_question":
        return incoming_history[-4:]
    if classification == "ambiguous":
        return incoming_history[-2:]
    return incoming_history[-HISTORY_TURNS:] if HISTORY_TURNS > 0 else incoming_history


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

    incoming_history = _clean_history_turns(history or [])
    if HISTORY_TURNS > 0:
        incoming_history = incoming_history[-HISTORY_TURNS:]

    confidentiality_key = (confidentiality or "SCM").strip().upper()
    if confidentiality_key not in SUPPORTED_CONFIDENTIALITY:
        confidentiality_key = "SCM"

    retrieval_profile = _resolve_retrieval_profile(mode, top_k)
    hits, retrieval_debug = _retrieve_with_history_awareness(
        query=query,
        incoming_history=incoming_history,
        retrieval_profile=retrieval_profile,
        confidentiality_key=confidentiality_key,
    )
    generation_history = _select_generation_history(incoming_history, retrieval_debug)

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
            "snippet": " | ".join(context_parts) + "\n" + h["chunk"],
        })

    cohere_history = []
    for turn in generation_history:
        role = (turn or {}).get("role", "")
        content = (turn or {}).get("content", "")
        if not content:
            continue
        role_lower = role.lower()
        if role_lower == "user":
            cohere_history.append({"role": "USER", "message": content})
        elif role_lower == "assistant":
            cohere_history.append({"role": "CHATBOT", "message": content})

    if not documents:
        documents = [{
            "title": "No matching knowledge found",
            "snippet": "I couldn’t find a matching project document for this question. I’ll answer using general Oracle Fusion guidance, and I’ll call out what would need confirmation from the relevant setup workbook, mapping file, design document, or implementation document.",
        }]

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

    llm_output = _enforce_citations(llm_output, hits)

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
        "history_length": len(generation_history),
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
            "query_classification": retrieval_debug["classification"],
            "retrieval_queries": retrieval_debug["candidate_queries"],
            "winning_query": retrieval_debug["winning_query"],
            "relevance_passed": retrieval_debug["relevance_passed"],
            "generation_history_used": len(generation_history),
        },
    }