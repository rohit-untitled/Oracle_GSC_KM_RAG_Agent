import os
import json
import logging
from typing import Dict, Any, List, Optional
import oracledb
from app.services.secure_config import require_env, get_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# WALLET_PATH = r"C:\Users\shshrohi\Desktop\KM_Docs_Rag_Agent\backend\Wallet_POCSOLUTIONSATPDEV_Nov_2025"
# os.environ["TNS_ADMIN"] = WALLET_PATH

WALLET_PATH = get_env("ORACLE_WALLET_PATH")
if WALLET_PATH:
    os.environ["TNS_ADMIN"] = WALLET_PATH


# DB_USER = os.getenv("ORACLE_DB_USER", "gsc_km_2")
# DB_PASSWORD = os.getenv("ORACLE_DB_PASSWORD", "Pa$$word#234")
# DB_TNS = os.getenv("ORACLE_DB_TNS", "pocsolutionsatpdev_high")

# VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1536"))

DB_USER = require_env("ORACLE_DB_USER")
DB_PASSWORD = require_env("ORACLE_DB_PASSWORD")
DB_TNS = require_env("ORACLE_DB_TNS")

VECTOR_DIM = int(get_env("VECTOR_DIM", "1536"))

_pool: Optional[oracledb.SessionPool] = None

def init_oracle_client():
    try:
        # IMPORTANT: MUST pass config_dir
        oracledb.init_oracle_client(config_dir=WALLET_PATH)
        logger.info(f"Oracle client initialized in THICK mode using wallet at: {WALLET_PATH}")
    except Exception as e:
        logger.error(f"Oracle client initialization failed: {e}")
        logger.info("Falling back to THIN mode (no wallet).")

init_oracle_client()

# Connection Pool
def get_pool() -> oracledb.SessionPool:
    global _pool
    if _pool is None:
        logger.info("Initializing Oracle SessionPool...")
        _pool = oracledb.SessionPool(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=DB_TNS,
            min=1,
            max=10,
            increment=1,
            encoding="UTF-8",
            threaded=True,
            getmode=oracledb.SPOOL_ATTRVAL_WAIT,
        )
        logger.info("Oracle SessionPool created.")
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


# Test connection
def test_connection() -> Dict[str, Any]:
    try:
        conn = oracledb.connect(DB_USER, DB_PASSWORD, DB_TNS)
        cur = conn.cursor()
        cur.execute("SELECT USER FROM dual")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return {"ok": True, "user": row[0]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# Insert one embedding
def insert_embedding_record(chunk_text: str, embedding_vector: List[float], metadata: Dict[str, Any]):

    conn = get_connection()
    cur = conn.cursor()

    metadata_json = json.dumps(metadata)
    embedding_string = "[" + ",".join(map(str, embedding_vector)) + "]"

    try:
        cur.execute("""
            INSERT INTO ai_vector_store (chunk, embedding, metadata)
            VALUES (:chunk, TO_VECTOR(:embedding_string), :metadata)
        """, {
            "chunk": chunk_text,
            "embedding_string": embedding_string,
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

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sql = """
        INSERT INTO ai_vector_store (chunk, embedding, metadata)
        VALUES (:chunk, TO_VECTOR(:embedding_string), :metadata)
    """

    skipped = 0
    inserted = 0

    for entry in data:
        emb = entry.get("embedding")

        if not emb or not isinstance(emb, list):
            logger.error(f"Skipping invalid embedding: {entry}")
            skipped += 1
            continue

        embedding_string = "[" + ",".join(map(str, emb)) + "]"

        cur.execute(sql, {
            "chunk": entry["chunk"],
            "embedding_string": embedding_string,
            "metadata": json.dumps(entry.get("metadata", {}))
        })

        inserted += 1

    conn.commit()
    cur.close()
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
