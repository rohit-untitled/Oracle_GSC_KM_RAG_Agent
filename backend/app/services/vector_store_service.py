import os
import json
import logging
from typing import Dict, Any, List, Optional

import oracledb
from app.services.secure_config import require_env, get_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DB_USER = require_env("ORACLE_DB_USER")
DB_PASSWORD = require_env("ORACLE_DB_PASSWORD")
DB_TNS = require_env("ORACLE_DB_TNS")

WALLET_PATH = get_env("ORACLE_WALLET_PATH")
ORACLE_MODE = get_env("ORACLE_MODE", "thin")
INSTANT_CLIENT_PATH = get_env("ORACLE_INSTANT_CLIENT")

VECTOR_DIM = int(get_env("VECTOR_DIM", "1536"))

def init_oracle():
    try:
        if WALLET_PATH:
            os.environ["TNS_ADMIN"] = WALLET_PATH
            logger.info(f"TNS_ADMIN set to {WALLET_PATH}")

        if ORACLE_MODE.lower() == "thick":
            if not INSTANT_CLIENT_PATH:
                raise RuntimeError("ORACLE_INSTANT_CLIENT not set for THICK mode")

            oracledb.init_oracle_client(
                lib_dir=INSTANT_CLIENT_PATH,
                config_dir=WALLET_PATH if WALLET_PATH else None,
            )
            logger.info("Oracle initialized in THICK mode")
        else:
            logger.info("Oracle running in THIN mode")

    except Exception as e:
        logger.error(f"Oracle initialization failed: {e}")
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
    return get_pool().acquire()


def close_pool():
    global _pool
    if _pool:
        try:
            _pool.close()
        except Exception as e:
            logger.exception("Error closing pool: %s", e)
        finally:
            _pool = None

# Insert one embedding
def insert_embedding_record(chunk_text: str, embedding_vector: List[float], metadata: Dict[str, Any]):

    conn = get_connection()
    cur = conn.cursor()

    metadata_json = json.dumps(metadata)
    embedding_string = "[" + ",".join(map(str, embedding_vector)) + "]"

    try:
        cur.setinputsizes(
            embedding_string=oracledb.DB_TYPE_VARCHAR,
            chunk=oracledb.DB_TYPE_CLOB,
            metadata=oracledb.DB_TYPE_CLOB
        )
        cur.execute("""
            INSERT INTO ai_vector_store (embedding, chunk, metadata)
            VALUES (TO_VECTOR(:embedding_string), :chunk, :metadata)
        """, {
            "embedding_string": embedding_string,
            "chunk": chunk_text,
            "metadata": metadata_json
        })

        conn.commit()
        logger.info("Inserted embedding successfully.")

    finally:
        cur.close()
        get_pool().release(conn)


# Insert multiple embeddings

def insert_embeddings_from_json(json_file_path: str):
    conn = get_connection()
    cur = conn.cursor()
    cur_check = conn.cursor()

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sql = """
        INSERT INTO ai_vector_store (embedding, chunk, metadata)
        VALUES (TO_VECTOR(:1), :2, :3)
    """

    skipped = 0
    inserted = 0
    rows = []

    cur.setinputsizes(
        oracledb.DB_TYPE_VARCHAR,
        oracledb.DB_TYPE_CLOB,
        oracledb.DB_TYPE_CLOB
    )

    def _chunk_exists(chunk_id: str) -> bool:
        if not chunk_id:
            return False
        cur_check.execute(
            """
            SELECT 1
            FROM ai_vector_store
            WHERE JSON_VALUE(metadata, '$.chunk_id') = :1
            FETCH FIRST 1 ROWS ONLY
            """,
            (chunk_id,),
        )
        return cur_check.fetchone() is not None

    for entry in data:
        emb = entry.get("embedding")

        if not emb or not isinstance(emb, list):
            logger.error(f"Skipping invalid embedding: {entry}")
            skipped += 1
            continue

        embedding_string = "[" + ",".join(map(str, emb)) + "]"

        metadata = {}
        if isinstance(entry.get("metadata"), dict):
            metadata.update(entry["metadata"])

        metadata.update({
            "chunk_id": entry.get("chunk_id"),
            "source_file": entry.get("source_file"),
            "chunk_index": entry.get("chunk_index"),
            "heading": entry.get("heading"),
        })

        if _chunk_exists(metadata.get("chunk_id")):
            skipped += 1
            continue

        rows.append((
            embedding_string,
            entry.get("chunk", ""),
            json.dumps(metadata)
        ))
        inserted += 1

    if rows:
        cur.executemany(sql, rows)
    conn.commit()
    cur.close()
    cur_check.close()
    get_pool().release(conn)

    logger.info(f"Batch insert done — Inserted={inserted}, Skipped={skipped}")

    return inserted    

# Vector Search

def search_similar_chunks(query_embedding: List[float], top_k: int = 5) -> List[dict]:

    conn = get_connection()
    cur = conn.cursor()

    embedding_string = "[" + ",".join(map(str, query_embedding)) + "]"

    sql = f"""
        SELECT chunk, metadata
        FROM ai_vector_store
        ORDER BY embedding <=> TO_VECTOR(:embedding_string)
        FETCH FIRST :top_k ROWS ONLY
    """

    cur.execute(sql, {
        "embedding_string": embedding_string,
        "top_k": top_k
    })

    hits = []

    for chunk, metadata_json in cur:

        if hasattr(chunk, "read"):
            chunk = chunk.read()

        if hasattr(metadata_json, "read"):
            metadata_json = metadata_json.read()

        metadata_json = metadata_json or "{}"

        try:
            metadata_dict = json.loads(metadata_json)
        except Exception:
            metadata_dict = {}

        hits.append({
            "chunk": chunk,
            "metadata": metadata_dict
        })


    cur.close()
    get_pool().release(conn)

    return hits

