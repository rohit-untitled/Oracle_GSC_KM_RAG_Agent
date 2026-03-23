from typing import List

from app.services.vector_store_service import (
    TABLE_CHUNK,
    TABLE_VECTOR,
    _read_lob_if_needed,
    close_connection,
    get_connection,
)


def search_similar_chunks(query_embedding: List[float], top_k: int = 5) -> List[dict]:
    conn = get_connection()
    cur = conn.cursor()

    embedding_string = "[" + ",".join(map(str, query_embedding)) + "]"

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
            v.RUN_ID
        FROM {TABLE_VECTOR} v
        JOIN {TABLE_CHUNK} c
          ON c.CHUNK_ID = v.CHUNK_ID
        LEFT JOIN XXGSC_KM_DOCUMENTS d
          ON d.DOCUMENT_ID = c.DOCUMENT_ID
        WHERE c.CHUNK_STATUS = 'ACTIVE'
        ORDER BY v.EMBEDDING_VECTOR <=> TO_VECTOR(:embedding_string)
        FETCH FIRST :top_k ROWS ONLY
    """

    cur.execute(sql, {"embedding_string": embedding_string, "top_k": top_k})

    hits = []
    for row in cur:
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
        ) = row

        chunk_text = _read_lob_if_needed(chunk_text)

        hits.append(
            {
                "chunk": chunk_text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "run_id": run_id,
                    "source_file": original_file_name or file_name,
                    "file_name": file_name,
                    "original_file_name": original_file_name,
                    "chunk_index": chunk_index,
                    "heading": section_heading,
                    "doc_type_code": doc_type_code,
                    "module_code": module_code,
                },
            }
        )

    cur.close()
    close_connection(conn)
    return hits