import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

import oracledb

from app.services.secure_config import get_env, require_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_USER = require_env("ORACLE_DB_USER")
DB_PASSWORD = require_env("ORACLE_DB_PASSWORD")
DB_TNS = require_env("ORACLE_DB_TNS")

WALLET_PATH = get_env("ORACLE_WALLET_PATH")
ORACLE_MODE = get_env("ORACLE_MODE", "thin")
INSTANT_CLIENT_PATH = get_env("ORACLE_INSTANT_CLIENT")

VECTOR_DIM = int(get_env("VECTOR_DIM", "1024"))

TABLE_CHUNK = "XXGSC_KM_DOCUMENT_CHUNK"
TABLE_VECTOR = "XXGSC_KM_CHUNK_VECTOR"
TABLE_CHUNK_METADATA = "XXGSC_KM_CHUNK_METADATA"


def init_oracle():
    try:
        mode = (ORACLE_MODE or "thin").strip().lower()

        if WALLET_PATH:
            os.environ["TNS_ADMIN"] = WALLET_PATH
            logger.info("TNS_ADMIN set to %s", WALLET_PATH)

        if mode == "thick":
            if not INSTANT_CLIENT_PATH:
                raise ValueError(
                    "ORACLE_MODE is 'thick' but ORACLE_INSTANT_CLIENT is not set"
                )

            oracledb.init_oracle_client(lib_dir=INSTANT_CLIENT_PATH)
            logger.info(
                "Oracle client initialized in thick mode | lib_dir=%s",
                INSTANT_CLIENT_PATH,
            )
        else:
            logger.info("Oracle client running in thin mode")
    except Exception as e:
        logger.error("Oracle initialization failed: %s", e)
        raise


init_oracle()

_pool: Optional[oracledb.ConnectionPool] = None


def get_pool() -> oracledb.ConnectionPool:
    global _pool

    if _pool is None:
        logger.info("Creating Oracle connection pool...")

        pool_kwargs = {
            "user": DB_USER,
            "password": DB_PASSWORD,
            "dsn": DB_TNS,
            "min": 1,
            "max": 10,
            "increment": 1,
            "getmode": oracledb.POOL_GETMODE_WAIT,
        }
        if WALLET_PATH:
            pool_kwargs["config_dir"] = WALLET_PATH

        _pool = oracledb.create_pool(**pool_kwargs)
        logger.info("Oracle connection pool created")

    return _pool


def get_connection() -> oracledb.Connection:
    resolved_wallet_path = os.path.abspath(WALLET_PATH) if WALLET_PATH else None
    logger.info(
        "Opening Oracle connection | user=%s dsn=%s config_dir=%s cwd=%s",
        DB_USER,
        DB_TNS,
        resolved_wallet_path,
        os.getcwd(),
    )

    return oracledb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        dsn=DB_TNS,
        config_dir=resolved_wallet_path,
    )


def close_pool():
    global _pool
    if _pool:
        try:
            _pool.close()
        except Exception as e:
            logger.exception("Error closing pool: %s", e)
        finally:
            _pool = None


def close_connection(conn: Optional[oracledb.Connection]) -> None:
    if conn:
        try:
            conn.close()
        except Exception as e:
            logger.exception("Error closing connection: %s", e)


def _make_chunk_id(entry: Dict[str, Any]) -> str:
    if entry.get("chunk_id"):
        return str(entry["chunk_id"])

    text = (entry.get("chunk") or "").strip()
    source_file = entry.get("source_file") or "UNKNOWN_SOURCE"
    chunk_index = entry.get("chunk_index") or 0
    seed = f"{source_file}|{chunk_index}|{text}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _make_chunk_hash(chunk_text: str) -> str:
    return hashlib.sha256((chunk_text or "").encode("utf-8")).hexdigest()


def _read_lob_if_needed(value: Any) -> Any:
    return value.read() if hasattr(value, "read") else value


def _serialize_metadata_json(metadata: Any) -> Optional[str]:
    if metadata is None:
        return None
    return json.dumps(metadata, ensure_ascii=False, default=str)


def _upsert_chunk_metadata(
    cur,
    *,
    chunk_id: str,
    document_id: str,
    run_id: str,
    document_type: Optional[str],
    source_file: Optional[str],
    page_number: Optional[int],
    sheet_name: Optional[str],
    row_number: Optional[int],
    metadata_json: Optional[str],
    created_by: str,
) -> None:
    cur.execute(
        f"""
        MERGE INTO {TABLE_CHUNK_METADATA} tgt
        USING (
            SELECT
                :chunk_id AS chunk_id,
                :document_id AS document_id,
                :run_id AS run_id,
                :document_type AS document_type,
                :source_file AS source_file,
                :page_number AS page_number,
                :sheet_name AS sheet_name,
                :row_number AS row_number,
                :metadata_json AS metadata_json,
                :created_by AS created_by
            FROM dual
        ) src
        ON (tgt.CHUNK_ID = src.chunk_id)
        WHEN MATCHED THEN UPDATE SET
            tgt.DOCUMENT_ID = src.document_id,
            tgt.RUN_ID = src.run_id,
            tgt.DOCUMENT_TYPE = src.document_type,
            tgt.SOURCE_FILE = src.source_file,
            tgt.PAGE_NUMBER = src.page_number,
            tgt.SHEET_NAME = src.sheet_name,
            tgt.ROW_NUMBER = src.row_number,
            tgt.METADATA_JSON = src.metadata_json,
            tgt.LAST_UPDATED_BY = src.created_by,
            tgt.LAST_UPDATE_DATE = CURRENT_TIMESTAMP
        WHEN NOT MATCHED THEN INSERT (
            METADATA_ID,
            CHUNK_ID,
            DOCUMENT_ID,
            RUN_ID,
            DOCUMENT_TYPE,
            SOURCE_FILE,
            PAGE_NUMBER,
            SHEET_NAME,
            ROW_NUMBER,
            METADATA_JSON,
            CREATED_BY,
            CREATION_DATE,
            LAST_UPDATED_BY,
            LAST_UPDATE_DATE
        ) VALUES (
            SYS_GUID(),
            src.chunk_id,
            src.document_id,
            src.run_id,
            src.document_type,
            src.source_file,
            src.page_number,
            src.sheet_name,
            src.row_number,
            src.metadata_json,
            src.created_by,
            CURRENT_TIMESTAMP,
            src.created_by,
            CURRENT_TIMESTAMP
        )
        """,
        {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "run_id": run_id,
            "document_type": document_type,
            "source_file": source_file,
            "page_number": page_number,
            "sheet_name": sheet_name,
            "row_number": row_number,
            "metadata_json": metadata_json,
            "created_by": created_by,
        },
    )


def insert_embedding_record(
    chunk_text: str,
    embedding_vector: List[float],
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    payload = {
        "chunk": chunk_text,
        "embedding": embedding_vector,
        "chunk_id": metadata.get("chunk_id"),
        "source_file": metadata.get("source_file"),
        "chunk_index": metadata.get("chunk_index"),
        "heading": metadata.get("heading"),
        "document_id": metadata.get("document_id"),
        "run_id": metadata.get("run_id"),
        "created_by": metadata.get("created_by", "KM_RAG_AGENT"),
    }
    return insert_embedding_payload(payload)


def insert_embedding_payload(entry: Dict[str, Any]) -> Dict[str, str]:
    emb = entry.get("embedding")
    if not emb or not isinstance(emb, list):
        raise ValueError("embedding must be a non-empty list")

    chunk_text = (entry.get("chunk") or "").strip()
    if not chunk_text:
        raise ValueError("chunk text is required")

    document_id = entry.get("document_id")
    run_id = entry.get("run_id")
    if not document_id or not run_id:
        raise ValueError("document_id and run_id are required for the new schema")

    chunk_id = _make_chunk_id(entry)
    chunk_hash = _make_chunk_hash(chunk_text)
    section_heading = entry.get("heading")
    chunk_index = entry.get("chunk_index") or 0
    created_by = entry.get("created_by", "KM_RAG_AGENT")
    embedding_string = "[" + ",".join(map(str, emb)) + "]"
    chunk_metadata = entry.get("metadata") or {}
    document_type = entry.get("document_type") or chunk_metadata.get("document_type")
    source_file = entry.get("source_file") or chunk_metadata.get("source_file") or chunk_metadata.get("file_name")
    page_number = entry.get("page_number") or chunk_metadata.get("page_number")
    sheet_name = entry.get("sheet_name") or chunk_metadata.get("sheet_name")
    row_number = entry.get("row_number") or chunk_metadata.get("row_number")
    metadata_json = _serialize_metadata_json(chunk_metadata)

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            f"""
            MERGE INTO {TABLE_CHUNK} tgt
            USING (
                SELECT
                    :chunk_id AS chunk_id,
                    :document_id AS document_id,
                    :run_id AS run_id,
                    :chunk_index AS chunk_index,
                    :section_heading AS section_heading,
                    :chunk_text AS chunk_text,
                    :chunk_hash AS chunk_hash,
                    :created_by AS created_by
                FROM dual
            ) src
            ON (tgt.CHUNK_ID = src.chunk_id)
            WHEN MATCHED THEN UPDATE SET
                tgt.DOCUMENT_ID = src.document_id,
                tgt.RUN_ID = src.run_id,
                tgt.CHUNK_INDEX = src.chunk_index,
                tgt.SECTION_HEADING = src.section_heading,
                tgt.CHUNK_TEXT = src.chunk_text,
                tgt.CHUNK_HASH = src.chunk_hash,
                tgt.LAST_UPDATED_BY = src.created_by,
                tgt.LAST_UPDATE_DATE = CURRENT_TIMESTAMP
            WHEN NOT MATCHED THEN INSERT (
                CHUNK_ID,
                DOCUMENT_ID,
                RUN_ID,
                CHUNK_INDEX,
                SECTION_HEADING,
                CHUNK_TEXT,
                CHUNK_HASH,
                IS_ANONYMIZED,
                CHUNK_STATUS,
                CREATED_BY,
                CREATION_DATE,
                LAST_UPDATED_BY,
                LAST_UPDATE_DATE
            ) VALUES (
                src.chunk_id,
                src.document_id,
                src.run_id,
                src.chunk_index,
                src.section_heading,
                src.chunk_text,
                src.chunk_hash,
                'Y',
                'ACTIVE',
                src.created_by,
                CURRENT_TIMESTAMP,
                src.created_by,
                CURRENT_TIMESTAMP
            )
            """,
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "run_id": run_id,
                "chunk_index": chunk_index,
                "section_heading": section_heading,
                "chunk_text": chunk_text,
                "chunk_hash": chunk_hash,
                "created_by": created_by,
            },
        )

        cur.execute(
            f"""
            MERGE INTO {TABLE_VECTOR} tgt
            USING (
                SELECT
                    :chunk_id AS chunk_id,
                    :document_id AS document_id,
                    :run_id AS run_id,
                    TO_VECTOR(:embedding_string) AS embedding_vector,
                    :created_by AS created_by
                FROM dual
            ) src
            ON (tgt.CHUNK_ID = src.chunk_id)
            WHEN MATCHED THEN UPDATE SET
                tgt.DOCUMENT_ID = src.document_id,
                tgt.RUN_ID = src.run_id,
                tgt.EMBEDDING_VECTOR = src.embedding_vector,
                tgt.LAST_UPDATED_BY = src.created_by,
                tgt.LAST_UPDATE_DATE = CURRENT_TIMESTAMP
            WHEN NOT MATCHED THEN INSERT (
                VECTOR_ID,
                CHUNK_ID,
                DOCUMENT_ID,
                RUN_ID,
                EMBEDDING_VECTOR,
                CREATED_BY,
                CREATION_DATE,
                LAST_UPDATED_BY,
                LAST_UPDATE_DATE
            ) VALUES (
                RAWTOHEX(SYS_GUID()),
                src.chunk_id,
                src.document_id,
                src.run_id,
                src.embedding_vector,
                src.created_by,
                CURRENT_TIMESTAMP,
                src.created_by,
                CURRENT_TIMESTAMP
            )
            """,
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "run_id": run_id,
                "embedding_string": embedding_string,
                "created_by": created_by,
            },
        )

        _upsert_chunk_metadata(
            cur,
            chunk_id=chunk_id,
            document_id=document_id,
            run_id=run_id,
            document_type=document_type,
            source_file=source_file,
            page_number=page_number,
            sheet_name=sheet_name,
            row_number=row_number,
            metadata_json=metadata_json,
            created_by=created_by,
        )

        conn.commit()
        return {"chunk_id": chunk_id, "status": "upserted"}
    finally:
        cur.close()
        close_connection(conn)


def insert_embeddings_from_json(json_file_path: str) -> int:
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    inserted = 0
    skipped = 0

    for entry in data:
        try:
            insert_embedding_payload(entry)
            inserted += 1
        except ValueError as exc:
            logger.warning("Skipping embedding row due to validation error: %s | row=%s", exc, entry)
            skipped += 1

    logger.info("Embedding sync done — Upserted=%s, Skipped=%s", inserted, skipped)
    return inserted


def insert_document_embeddings(
    document_id: str,
    run_id: str,
    chunks: List[Dict[str, Any]],
    created_by: str = "KM_RAG_AGENT",
) -> int:
    inserted = 0

    for idx, entry in enumerate(chunks):
        payload = {
            "document_id": document_id,
            "run_id": run_id,
            "chunk_id": entry.get("chunk_id"),
            "chunk_index": entry.get("chunk_index", idx),
            "heading": entry.get("heading"),
            "source_file": entry.get("source_file"),
            "chunk": entry.get("chunk", ""),
            "embedding": entry.get("embedding"),
            "document_type": entry.get("document_type"),
            "sheet_name": entry.get("sheet_name"),
            "row_number": entry.get("row_number"),
            "page_number": entry.get("page_number"),
            "metadata": entry.get("metadata"),
            "created_by": created_by,
        }
        insert_embedding_payload(payload)
        inserted += 1

    return inserted


def search_similar_chunks(query_embedding: List[float], top_k: int = 12) -> List[dict]:
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