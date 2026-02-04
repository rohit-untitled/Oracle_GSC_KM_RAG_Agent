## for windows oracle client

# import os
# import json
# import logging
# from typing import Dict, Any, List, Optional
# import oracledb
# from app.services.secure_config import require_env, get_env

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# WALLET_PATH = get_env("ORACLE_WALLET_PATH")
# if WALLET_PATH:
#     os.environ["TNS_ADMIN"] = WALLET_PATH

# DB_USER = require_env("ORACLE_DB_USER")
# DB_PASSWORD = require_env("ORACLE_DB_PASSWORD")
# DB_TNS = require_env("ORACLE_DB_TNS")

# VECTOR_DIM = int(get_env("VECTOR_DIM", "1536"))

# _pool: Optional[oracledb.SessionPool] = None

# def init_oracle_client():
#     try:
#         lib_path = "/opt/oracle/instantclient/instantclient_23_26"

#         if os.path.exists(lib_path):
#             oracledb.init_oracle_client(lib_dir=lib_path)
#             logger.info(f"Oracle client initialized in THICK mode using lib_dir={lib_path}")
#         elif WALLET_PATH:
#             oracledb.init_oracle_client(config_dir=WALLET_PATH)
#             logger.info(f"Oracle client initialized with wallet config_dir={WALLET_PATH}")

#         else:
#             logger.info("Instant Client not found. Running in THIN mode (no client needed).")

#     except Exception as e:
#         logger.error(f"Oracle client initialization failed: {e}")
#         logger.info("Falling back to THIN mode.")

# init_oracle_client()

# # Connection Pool
# def get_pool() -> oracledb.SessionPool:
#     global _pool
#     if _pool is None:
#         logger.info("Initializing Oracle SessionPool...")
#         _pool = oracledb.SessionPool(
#             user=DB_USER,
#             password=DB_PASSWORD,
#             dsn=DB_TNS,
#             min=1,
#             max=10,
#             increment=1,
#             encoding="UTF-8",
#             threaded=True,
#             getmode=oracledb.SPOOL_ATTRVAL_WAIT,
#         )
#         logger.info("Oracle SessionPool created.")
#     return _pool


# def get_connection() -> oracledb.Connection:
#     return get_pool().acquire()


# def close_pool():
#     global _pool
#     if _pool:
#         try:
#             _pool.close()
#         except Exception as e:
#             logger.exception("Error closing pool: %s", e)
#         finally:
#             _pool = None


# # Test connection
# def test_connection() -> Dict[str, Any]:
#     try:
#         conn = oracledb.connect(DB_USER, DB_PASSWORD, DB_TNS)
#         cur = conn.cursor()
#         cur.execute("SELECT USER FROM dual")
#         row = cur.fetchone()
#         cur.close()
#         conn.close()
#         return {"ok": True, "user": row[0]}
#     except Exception as e:
#         return {"ok": False, "error": str(e)}


# # Insert one embedding
# def insert_embedding_record(chunk_text: str, embedding_vector: List[float], metadata: Dict[str, Any]):

#     conn = get_connection()
#     cur = conn.cursor()

#     metadata_json = json.dumps(metadata)
#     embedding_string = "[" + ",".join(map(str, embedding_vector)) + "]"

#     try:
#         cur.execute("""
#             INSERT INTO ai_vector_store (chunk, embedding, metadata)
#             VALUES (:chunk, TO_VECTOR(:embedding_string), :metadata)
#         """, {
#             "chunk": chunk_text,
#             "embedding_string": embedding_string,
#             "metadata": metadata_json
#         })

#         conn.commit()
#         logger.info("Inserted embedding successfully.")

#     finally:
#         cur.close()
#         get_pool().release(conn)


# # Insert multiple embeddings

# def insert_embeddings_from_json(json_file_path: str):
#     conn = get_connection()
#     cur = conn.cursor()

#     with open(json_file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     sql = """
#         INSERT INTO ai_vector_store (chunk, embedding, metadata)
#         VALUES (:chunk, TO_VECTOR(:embedding_string), :metadata)
#     """

#     skipped = 0
#     inserted = 0

#     for entry in data:
#         emb = entry.get("embedding")

#         if not emb or not isinstance(emb, list):
#             logger.error(f"Skipping invalid embedding: {entry}")
#             skipped += 1
#             continue

#         embedding_string = "[" + ",".join(map(str, emb)) + "]"

#         cur.execute(sql, {
#             "chunk": entry["chunk"],
#             "embedding_string": embedding_string,
#             "metadata": json.dumps(entry.get("metadata", {}))
#         })

#         inserted += 1

#     conn.commit()
#     cur.close()
#     get_pool().release(conn)

#     logger.info(f"Batch insert done — Inserted={inserted}, Skipped={skipped}")

#     return inserted    



# # Vector Search

# def search_similar_chunks(query_embedding: List[float], top_k: int = 5) -> List[dict]:

#     conn = get_connection()
#     cur = conn.cursor()

#     embedding_string = "[" + ",".join(map(str, query_embedding)) + "]"

#     sql = f"""
#         SELECT chunk, metadata
#         FROM ai_vector_store
#         ORDER BY embedding <=> TO_VECTOR(:embedding_string)
#         FETCH FIRST :top_k ROWS ONLY
#     """

#     cur.execute(sql, {
#         "embedding_string": embedding_string,
#         "top_k": top_k
#     })

#     hits = []

#     for chunk, metadata_json in cur:

#         if hasattr(chunk, "read"):
#             chunk = chunk.read()

#         if hasattr(metadata_json, "read"):
#             metadata_json = metadata_json.read()

#         metadata_json = metadata_json or "{}"

#         try:
#             metadata_dict = json.loads(metadata_json)
#         except Exception:
#             metadata_dict = {}

#         hits.append({
#             "chunk": chunk,
#             "metadata": metadata_dict
#         })


#     cur.close()
#     get_pool().release(conn)

#     return hits



#for VM client

import os
import json
import logging
from typing import Dict, Any, List, Optional

import oracledb
from app.services.secure_config import require_env, get_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# Environment
# --------------------------------------------------
DB_USER = require_env("ORACLE_DB_USER")
DB_PASSWORD = require_env("ORACLE_DB_PASSWORD")
DB_TNS = require_env("ORACLE_DB_TNS")
WALLET_PATH = require_env("ORACLE_WALLET_PATH")
VECTOR_DIM = int(get_env("VECTOR_DIM", "1536"))

# --------------------------------------------------
# Force THICK mode with Wallet
# --------------------------------------------------
INSTANT_CLIENT = "/opt/oracle/instantclient_23_26"

if not os.path.exists(INSTANT_CLIENT):
    raise RuntimeError("Oracle Instant Client not found")

if not os.path.exists(WALLET_PATH):
    raise RuntimeError("Oracle Wallet path not found")

oracledb.init_oracle_client(
    lib_dir=INSTANT_CLIENT,
    config_dir=WALLET_PATH
)

# --------------------------------------------------
# Connection Pool
# --------------------------------------------------
_pool: Optional[oracledb.ConnectionPool] = None


def get_pool() -> oracledb.ConnectionPool:
    global _pool
    if _pool is None:
        logger.info("Creating Oracle connection pool...")
        _pool = oracledb.create_pool(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=DB_TNS,
            min=1,
            max=5,
            increment=1,
            timeout=60,
            getmode=oracledb.POOL_GETMODE_WAIT,
        )
        logger.info("Oracle connection pool created")
    return _pool


def get_connection() -> oracledb.Connection:
    return get_pool().acquire()


def close_pool():
    global _pool
    if _pool:
        try:
            _pool.close()
            logger.info("Oracle pool closed")
        finally:
            _pool = None


# --------------------------------------------------
# Health Check
# --------------------------------------------------
def test_connection() -> Dict[str, Any]:
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT USER FROM dual")
        user = cur.fetchone()[0]
        return {"ok": True, "user": user}
    except Exception as e:
        logger.exception("DB connection test failed")
        return {"ok": False, "error": str(e)}
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()  # returns to pool


# --------------------------------------------------
# Inserts
# --------------------------------------------------
def insert_embedding_record(
    chunk_text: str,
    embedding_vector: List[float],
    metadata: Dict[str, Any],
):
    conn = None
    cur = None

    embedding_string = "[" + ",".join(map(str, embedding_vector)) + "]"
    metadata_json = json.dumps(metadata)

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ai_vector_store (chunk, embedding, metadata)
            VALUES (:chunk, TO_VECTOR(:embedding), :metadata)
            """,
            {
                "chunk": chunk_text,
                "embedding": embedding_string,
                "metadata": metadata_json,
            },
        )
        conn.commit()
        logger.info("Inserted embedding")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def insert_embeddings_from_json(json_file_path: str) -> int:
    conn = None
    cur = None
    inserted = 0

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        conn = get_connection()
        cur = conn.cursor()

        sql = """
            INSERT INTO ai_vector_store (chunk, embedding, metadata)
            VALUES (:chunk, TO_VECTOR(:embedding), :metadata)
        """

        for entry in data:
            emb = entry.get("embedding")
            if not isinstance(emb, list):
                continue

            embedding_string = "[" + ",".join(map(str, emb)) + "]"

            cur.execute(
                sql,
                {
                    "chunk": entry["chunk"],
                    "embedding": embedding_string,
                    "metadata": json.dumps(entry.get("metadata", {})),
                },
            )
            inserted += 1

        conn.commit()
        logger.info(f"Batch insert complete: {inserted} rows")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    return inserted


# --------------------------------------------------
# Vector Search
# --------------------------------------------------
def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:

    conn = None
    cur = None
    results = []

    embedding_string = "[" + ",".join(map(str, query_embedding)) + "]"

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT chunk, metadata
            FROM ai_vector_store
            ORDER BY embedding <=> TO_VECTOR(:embedding)
            FETCH FIRST :top_k ROWS ONLY
            """,
            {
                "embedding": embedding_string,
                "top_k": top_k,
            },
        )

        for chunk, metadata_json in cur:
            if hasattr(chunk, "read"):
                chunk = chunk.read()
            if hasattr(metadata_json, "read"):
                metadata_json = metadata_json.read()

            try:
                metadata = json.loads(metadata_json or "{}")
            except Exception:
                metadata = {}

            results.append(
                {
                    "chunk": chunk,
                    "metadata": metadata,
                }
            )

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    return results
