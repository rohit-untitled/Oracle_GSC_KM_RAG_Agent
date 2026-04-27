from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.services.vector_store_service import (
    TABLE_CHUNK,
    TABLE_VECTOR,
    _read_lob_if_needed,
    close_connection,
    get_connection,
)


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
    extra_metadata = json.loads(metadata_json) if metadata_json else {}

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
    try:
        sql = f"""
            SELECT
                c.CHUNK_ID,
                c.CHUNK_TEXT,
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
                m.METADATA_JSON
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
    return keywords


def _fetch_keyword_hits(
    conn,
    *,
    keywords: List[str],
    top_k: int,
    document_type: Optional[str],
    project_id: Optional[str],
    module_code: Optional[str],
    file_name: Optional[str],
) -> List[Dict[str, Any]]:
    if not keywords or top_k <= 0:
        return []

    cur = conn.cursor()
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
        if module_code:
            filters.append("d.MODULE_CODE = :module_code")
            binds["module_code"] = module_code
        if file_name:
            filters.append("(d.FILE_NAME = :file_name OR d.ORIGINAL_FILE_NAME = :file_name)")
            binds["file_name"] = file_name

        sql = f"""
            SELECT
                c.CHUNK_ID,
                c.CHUNK_TEXT,
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
                m.METADATA_JSON
            FROM {TABLE_CHUNK} c
            LEFT JOIN {TABLE_VECTOR} v
              ON v.CHUNK_ID = c.CHUNK_ID
            LEFT JOIN XXGSC_KM_DOCUMENTS d
              ON d.DOCUMENT_ID = c.DOCUMENT_ID
            LEFT JOIN XXGSC_KM_CHUNK_METADATA m
              ON m.CHUNK_ID = c.CHUNK_ID
            WHERE {' AND '.join(filters)}
            FETCH FIRST :fetch_rows ROWS ONLY
        """
        cur.execute(sql, binds)
        hits = [_normalize_hit(row) for row in cur]
        for hit in hits:
            hit.setdefault("metadata", {})["retrieval_mode"] = "keyword"
        return hits
    finally:
        cur.close()


def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = 8,
    *,
    query_text: Optional[str] = None,
    document_type: Optional[str] = None,
    project_id: Optional[str] = None,
    module_code: Optional[str] = None,
    file_name: Optional[str] = None,
    rerank_top_n: int = 20,
    neighbor_radius: int = 2,
    use_hybrid: bool = True,
) -> List[dict]:
    conn = get_connection()
    cur = conn.cursor()

    embedding_string = "[" + ",".join(map(str, query_embedding)) + "]"

    filters = ["c.CHUNK_STATUS = 'ACTIVE'"]
    binds: Dict[str, Any] = {
        "embedding_string": embedding_string,
        "fetch_rows": max(top_k, rerank_top_n),
    }

    if document_type:
        filters.append("m.DOCUMENT_TYPE = :document_type")
        binds["document_type"] = document_type
    if project_id:
        filters.append("d.PROJECT_ID = :project_id")
        binds["project_id"] = project_id
    if module_code:
        filters.append("d.MODULE_CODE = :module_code")
        binds["module_code"] = module_code
    if file_name:
        filters.append("(d.FILE_NAME = :file_name OR d.ORIGINAL_FILE_NAME = :file_name)")
        binds["file_name"] = file_name

    sql = f"""
        SELECT
            c.CHUNK_ID,
            c.CHUNK_TEXT,
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
            m.METADATA_JSON
        FROM {TABLE_VECTOR} v
        JOIN {TABLE_CHUNK} c
          ON c.CHUNK_ID = v.CHUNK_ID
        LEFT JOIN XXGSC_KM_DOCUMENTS d
          ON d.DOCUMENT_ID = c.DOCUMENT_ID
        LEFT JOIN XXGSC_KM_CHUNK_METADATA m
          ON m.CHUNK_ID = c.CHUNK_ID
        WHERE {' AND '.join(filters)}
        ORDER BY v.EMBEDDING_VECTOR <=> TO_VECTOR(:embedding_string)
        FETCH FIRST :fetch_rows ROWS ONLY
    """

    cur.execute(sql, binds)
    candidate_hits = [_normalize_hit(row) for row in cur]
    for hit in candidate_hits:
        hit.setdefault("metadata", {})["retrieval_mode"] = "vector"
    cur.close()

    keyword_hits: List[Dict[str, Any]] = []
    if use_hybrid and query_text:
        keyword_hits = _fetch_keyword_hits(
            conn,
            keywords=_extract_keywords(query_text),
            top_k=max(top_k, rerank_top_n),
            document_type=document_type,
            project_id=project_id,
            module_code=module_code,
            file_name=file_name,
        )

    merged_candidates: Dict[str, Dict[str, Any]] = {}
    for hit in candidate_hits + keyword_hits:
        chunk_id = (hit.get("metadata", {}) or {}).get("chunk_id")
        if not chunk_id:
            continue
        existing = merged_candidates.get(chunk_id)
        if existing is None:
            merged_candidates[chunk_id] = hit
            continue

        existing_mode = (existing.get("metadata", {}) or {}).get("retrieval_mode", "")
        new_mode = (hit.get("metadata", {}) or {}).get("retrieval_mode", "")
        if existing_mode != new_mode:
            merged_candidates[chunk_id].setdefault("metadata", {})["retrieval_mode"] = "hybrid"

    reranked_hits = list(merged_candidates.values())[:top_k]

    expanded_hits: List[Dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for hit in reranked_hits:
        metadata = hit.get("metadata", {}) or {}
        document_id_value = metadata.get("document_id")
        chunk_index_value = metadata.get("chunk_index")
        if document_id_value is None or chunk_index_value is None:
            chunk_id = metadata.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                expanded_hits.append(hit)
                seen_chunk_ids.add(chunk_id)
            continue

        neighbors = _fetch_neighbor_chunks(
            conn,
            document_id=str(document_id_value),
            chunk_index=int(chunk_index_value),
            radius=neighbor_radius,
        )
        for neighbor in neighbors:
            chunk_id = (neighbor.get("metadata", {}) or {}).get("chunk_id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                expanded_hits.append(neighbor)
                seen_chunk_ids.add(chunk_id)

    close_connection(conn)
    return expanded_hits