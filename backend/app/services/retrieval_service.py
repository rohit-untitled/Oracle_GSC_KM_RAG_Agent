from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.services.secure_config import get_env
from app.services.vector_store_service import (
    TABLE_CHUNK,
    TABLE_VECTOR,
    _read_lob_if_needed,
    close_connection,
    fetch_lobs_as_strings,
    get_connection,
)

logger = logging.getLogger(__name__)

MAX_KEYWORD_TERMS = int(get_env("RETRIEVAL_MAX_KEYWORD_TERMS", "6"))
LOB_SUBSTR_CHARS = min(4000, max(1, int(get_env("RETRIEVAL_LOB_SUBSTR_CHARS", "4000"))))
DB_CALL_TIMEOUT_MS = int(get_env("RETRIEVAL_DB_CALL_TIMEOUT_MS", "45000"))


def _normalize_hit(row: tuple[Any, ...]) -> Dict[str, Any]:
    (
        chunk_id,
        chunk_text,
        section_heading,
        chunk_index,
        document_id,
        file_name,
        original_file_name,
        doc_type_code,
        module_code,
        run_id,
        document_type,
        source_file,
        page_number,
        sheet_name,
        row_number,
        metadata_json,
    ) = row

    chunk_text = _read_lob_if_needed(chunk_text)
    metadata_json = _read_lob_if_needed(metadata_json)
    try:
        extra_metadata = json.loads(metadata_json) if metadata_json else {}
    except (TypeError, json.JSONDecodeError):
        extra_metadata = {}

    metadata = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "run_id": run_id,
        "source_file": source_file or original_file_name or file_name,
        "file_name": file_name,
        "original_file_name": original_file_name,
        "chunk_index": chunk_index,
        "heading": section_heading,
        "doc_type_code": doc_type_code,
        "module_code": module_code,
        "document_type": document_type,
        "page_number": page_number,
        "sheet_name": sheet_name,
        "row_number": row_number,
    }
    metadata.update(extra_metadata)

    return {
        "chunk": chunk_text,
        "metadata": metadata,
    }


def _fetch_neighbor_chunks(
    conn,
    *,
    document_id: str,
    chunk_index: int,
    radius: int,
) -> List[Dict[str, Any]]:
    if radius <= 0:
        return []

    cur = conn.cursor()
    fetch_lobs_as_strings(cur)
    try:
        sql = f"""
            SELECT
                c.CHUNK_ID,
                DBMS_LOB.SUBSTR(c.CHUNK_TEXT, {LOB_SUBSTR_CHARS}, 1) AS CHUNK_TEXT,
                c.SECTION_HEADING,
                c.CHUNK_INDEX,
                c.DOCUMENT_ID,
                d.FILE_NAME,
                d.ORIGINAL_FILE_NAME,
                d.DOC_TYPE_CODE,
                d.MODULE_CODE,
                v.RUN_ID,
                m.DOCUMENT_TYPE,
                m.SOURCE_FILE,
                m.PAGE_NUMBER,
                m.SHEET_NAME,
                m.ROW_NUMBER,
                DBMS_LOB.SUBSTR(m.METADATA_JSON, {LOB_SUBSTR_CHARS}, 1) AS METADATA_JSON
            FROM {TABLE_CHUNK} c
            LEFT JOIN {TABLE_VECTOR} v
              ON v.CHUNK_ID = c.CHUNK_ID
            LEFT JOIN XXGSC_KM_DOCUMENTS d
              ON d.DOCUMENT_ID = c.DOCUMENT_ID
            LEFT JOIN XXGSC_KM_CHUNK_METADATA m
              ON m.CHUNK_ID = c.CHUNK_ID
            WHERE c.CHUNK_STATUS = 'ACTIVE'
              AND c.DOCUMENT_ID = :document_id
              AND c.CHUNK_INDEX BETWEEN :min_index AND :max_index
            ORDER BY c.CHUNK_INDEX
        """
        cur.execute(
            sql,
            {
                "document_id": document_id,
                "min_index": max(0, chunk_index - radius),
                "max_index": chunk_index + radius,
            },
        )
        return [_normalize_hit(row) for row in cur]
    except Exception:
        logger.warning(
            "Neighbor chunk fetch failed | document_id=%s chunk_index=%s radius=%s",
            document_id,
            chunk_index,
            radius,
            exc_info=True,
        )
        return []
    finally:
        cur.close()


def _extract_keywords(query_text: str, min_len: int = 3) -> List[str]:
    seen: set[str] = set()
    keywords: List[str] = []
    for token in re.findall(r"[A-Za-z0-9_\-]+", query_text or ""):
        normalized = token.strip().lower()
        if len(normalized) < min_len:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        keywords.append(normalized)
        if len(keywords) >= MAX_KEYWORD_TERMS:
            break
    return keywords


def _apply_call_timeout(conn) -> None:
    if DB_CALL_TIMEOUT_MS <= 0:
        return
    try:
        conn.call_timeout = DB_CALL_TIMEOUT_MS
    except Exception:
        logger.debug("Oracle connection does not support call_timeout", exc_info=True)


def _fetch_keyword_hits(
    conn,
    *,
    keywords: List[str],
    top_k: int,
    document_type: Optional[str],
    project_id: Optional[str],
    module_code: Optional[str],
    confidentiality: Optional[str],
    file_name: Optional[str],
) -> List[Dict[str, Any]]:
    if not keywords or top_k <= 0:
        return []

    cur = conn.cursor()
    fetch_lobs_as_strings(cur)
    try:
        filters = ["c.CHUNK_STATUS = 'ACTIVE'"]
        binds: Dict[str, Any] = {"fetch_rows": top_k}

        keyword_clauses = []
        for idx, keyword in enumerate(keywords):
            bind_key = f"kw_{idx}"
            keyword_clauses.append(f"LOWER(c.CHUNK_TEXT) LIKE :{bind_key}")
            binds[bind_key] = f"%{keyword}%"

        if keyword_clauses:
            filters.append("(" + " OR ".join(keyword_clauses) + ")")

        if document_type:
            filters.append("m.DOCUMENT_TYPE = :document_type")
            binds["document_type"] = document_type
        if project_id:
            filters.append("d.PROJECT_ID = :project_id")
            binds["project_id"] = project_id
        if confidentiality:
            filters.append("p.CONFIDENTIALITY = :confidentiality")
            binds["confidentiality"] = confidentiality
        if module_code:
            filters.append("d.MODULE_CODE = :module_code")
            binds["module_code"] = module_code
        if file_name:
            filters.append("(d.FILE_NAME = :file_name OR d.ORIGINAL_FILE_NAME = :file_name)")
            binds["file_name"] = file_name

        sql = f"""
            SELECT
                c.CHUNK_ID,
                DBMS_LOB.SUBSTR(c.CHUNK_TEXT, {LOB_SUBSTR_CHARS}, 1) AS CHUNK_TEXT,
                c.SECTION_HEADING,
                c.CHUNK_INDEX,
                c.DOCUMENT_ID,
                d.FILE_NAME,
                d.ORIGINAL_FILE_NAME,
                d.DOC_TYPE_CODE,
                d.MODULE_CODE,
                v.RUN_ID,
                m.DOCUMENT_TYPE,
                m.SOURCE_FILE,
                m.PAGE_NUMBER,
                m.SHEET_NAME,
                m.ROW_NUMBER,
                DBMS_LOB.SUBSTR(m.METADATA_JSON, {LOB_SUBSTR_CHARS}, 1) AS METADATA_JSON
            FROM {TABLE_CHUNK} c
            LEFT JOIN {TABLE_VECTOR} v
              ON v.CHUNK_ID = c.CHUNK_ID
            LEFT JOIN XXGSC_KM_DOCUMENTS d
              ON d.DOCUMENT_ID = c.DOCUMENT_ID
            LEFT JOIN XXGSC_KM_PROJECTS p
              ON p.PROJECT_ID = d.PROJECT_ID
            LEFT JOIN XXGSC_KM_CHUNK_METADATA m
              ON m.CHUNK_ID = c.CHUNK_ID
            WHERE {' AND '.join(filters)}
            ORDER BY c.CHUNK_ID
            FETCH FIRST :fetch_rows ROWS ONLY
        """
        cur.execute(sql, binds)
        hits = [_normalize_hit(row) for row in cur]
        for rank, hit in enumerate(hits, start=1):
            hit.setdefault("metadata", {})["retrieval_mode"] = "keyword"
            hit.setdefault("metadata", {})["keyword_rank"] = rank
        return hits
    except Exception:
        logger.warning(
            "Keyword retrieval failed; continuing with vector results | keywords=%s",
            keywords,
            exc_info=True,
        )
        return []
    finally:
        cur.close()


def _candidate_score(hit: Dict[str, Any]) -> float:
    metadata = (hit or {}).get("metadata", {}) or {}
    vector_rank = metadata.get("vector_rank")
    keyword_rank = metadata.get("keyword_rank")

    score = 0.0
    if isinstance(vector_rank, int):
        score += 1.0 / (vector_rank + 1)
    if isinstance(keyword_rank, int):
        score += 0.7 / (keyword_rank + 1)
    if metadata.get("retrieval_mode") == "hybrid":
        score += 0.5
    return score


def _merge_and_rank_candidates(
    *,
    candidate_hits: List[Dict[str, Any]],
    keyword_hits: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for hit in candidate_hits + keyword_hits:
        metadata = (hit or {}).get("metadata", {}) or {}
        chunk_id = metadata.get("chunk_id")
        if not chunk_id:
            continue

        existing = merged.get(chunk_id)
        if existing is None:
            merged[chunk_id] = hit
            continue

        existing_metadata = existing.setdefault("metadata", {})
        for key in ("vector_rank", "keyword_rank"):
            if key in metadata and key not in existing_metadata:
                existing_metadata[key] = metadata[key]

        existing_mode = existing_metadata.get("retrieval_mode", "")
        new_mode = metadata.get("retrieval_mode", "")
        if existing_mode != new_mode:
            existing_metadata["retrieval_mode"] = "hybrid"

    ranked = sorted(merged.values(), key=_candidate_score, reverse=True)
    return ranked[:top_k]


def _append_unique_hit(
    output: List[Dict[str, Any]],
    seen_chunk_ids: set[str],
    hit: Dict[str, Any],
) -> None:
    chunk_id = ((hit or {}).get("metadata", {}) or {}).get("chunk_id")
    if chunk_id and chunk_id not in seen_chunk_ids:
        output.append(hit)
        seen_chunk_ids.add(chunk_id)


def _expand_neighbors(
    *,
    conn,
    hits: List[Dict[str, Any]],
    neighbor_radius: int,
) -> List[Dict[str, Any]]:
    expanded_hits: List[Dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for hit in hits:
        metadata = hit.get("metadata", {}) or {}
        document_id_value = metadata.get("document_id")
        chunk_index_value = metadata.get("chunk_index")
        if document_id_value is None or chunk_index_value is None or neighbor_radius <= 0:
            _append_unique_hit(expanded_hits, seen_chunk_ids, hit)
            continue

        neighbors = _fetch_neighbor_chunks(
            conn,
            document_id=str(document_id_value),
            chunk_index=int(chunk_index_value),
            radius=neighbor_radius,
        )
        if not neighbors:
            _append_unique_hit(expanded_hits, seen_chunk_ids, hit)
            continue

        for neighbor in neighbors:
            _append_unique_hit(expanded_hits, seen_chunk_ids, neighbor)

    return expanded_hits


def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = 8,
    *,
    query_text: Optional[str] = None,
    document_type: Optional[str] = None,
    project_id: Optional[str] = None,
    module_code: Optional[str] = None,
    confidentiality: Optional[str] = None,
    file_name: Optional[str] = None,
    rerank_top_n: int = 20,
    neighbor_radius: int = 2,
    use_hybrid: bool = True,
) -> List[dict]:
    started_at = time.perf_counter()
    conn = get_connection()
    _apply_call_timeout(conn)
    try:
        cur = conn.cursor()
        fetch_lobs_as_strings(cur)

        embedding_string = "[" + ",".join(map(str, query_embedding)) + "]"

        filters = ["c.CHUNK_STATUS = 'ACTIVE'"]
        fetch_rows = max(1, min(max(top_k, rerank_top_n), 50))
        binds: Dict[str, Any] = {
            "embedding_string": embedding_string,
            "fetch_rows": fetch_rows,
        }

        if document_type:
            filters.append("m.DOCUMENT_TYPE = :document_type")
            binds["document_type"] = document_type
        if project_id:
            filters.append("d.PROJECT_ID = :project_id")
            binds["project_id"] = project_id
        if confidentiality:
            filters.append("p.CONFIDENTIALITY = :confidentiality")
            binds["confidentiality"] = confidentiality
        if module_code:
            filters.append("d.MODULE_CODE = :module_code")
            binds["module_code"] = module_code
        if file_name:
            filters.append("(d.FILE_NAME = :file_name OR d.ORIGINAL_FILE_NAME = :file_name)")
            binds["file_name"] = file_name

        sql = f"""
            SELECT
                c.CHUNK_ID,
                DBMS_LOB.SUBSTR(c.CHUNK_TEXT, {LOB_SUBSTR_CHARS}, 1) AS CHUNK_TEXT,
                c.SECTION_HEADING,
                c.CHUNK_INDEX,
                c.DOCUMENT_ID,
                d.FILE_NAME,
                d.ORIGINAL_FILE_NAME,
                d.DOC_TYPE_CODE,
                d.MODULE_CODE,
                v.RUN_ID,
                m.DOCUMENT_TYPE,
                m.SOURCE_FILE,
                m.PAGE_NUMBER,
                m.SHEET_NAME,
                m.ROW_NUMBER,
                DBMS_LOB.SUBSTR(m.METADATA_JSON, {LOB_SUBSTR_CHARS}, 1) AS METADATA_JSON
            FROM {TABLE_VECTOR} v
            JOIN {TABLE_CHUNK} c
              ON c.CHUNK_ID = v.CHUNK_ID
            LEFT JOIN XXGSC_KM_DOCUMENTS d
              ON d.DOCUMENT_ID = c.DOCUMENT_ID
            LEFT JOIN XXGSC_KM_PROJECTS p
              ON p.PROJECT_ID = d.PROJECT_ID
            LEFT JOIN XXGSC_KM_CHUNK_METADATA m
              ON m.CHUNK_ID = c.CHUNK_ID
            WHERE {' AND '.join(filters)}
            ORDER BY v.EMBEDDING_VECTOR <=> TO_VECTOR(:embedding_string)
            FETCH FIRST :fetch_rows ROWS ONLY
        """

        try:
            cur.execute(sql, binds)
            candidate_hits = [_normalize_hit(row) for row in cur]
        finally:
            cur.close()

        for rank, hit in enumerate(candidate_hits, start=1):
            metadata = hit.setdefault("metadata", {})
            metadata["retrieval_mode"] = "vector"
            metadata["vector_rank"] = rank

        keyword_hits: List[Dict[str, Any]] = []
        if use_hybrid and query_text:
            keyword_hits = _fetch_keyword_hits(
                conn,
                keywords=_extract_keywords(query_text),
                top_k=fetch_rows,
                document_type=document_type,
                project_id=project_id,
                module_code=module_code,
                confidentiality=confidentiality,
                file_name=file_name,
            )

        merged_candidates = _merge_and_rank_candidates(
            candidate_hits=candidate_hits,
            keyword_hits=keyword_hits,
            top_k=max(1, top_k),
        )

        expanded_hits = _expand_neighbors(
            conn=conn,
            hits=merged_candidates,
            neighbor_radius=max(0, neighbor_radius),
        )
        logger.info(
            "Similarity search complete | vector_hits=%s keyword_hits=%s returned=%s top_k=%s hybrid=%s elapsed=%.3fs",
            len(candidate_hits),
            len(keyword_hits),
            len(expanded_hits),
            top_k,
            use_hybrid,
            time.perf_counter() - started_at,
        )
        return expanded_hits
    finally:
        close_connection(conn)
