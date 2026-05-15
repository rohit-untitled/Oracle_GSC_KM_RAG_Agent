import logging
from typing import Any, Dict, List, Optional

import oracledb

from app.services.vector_store_service import close_connection, get_connection


logger = logging.getLogger(__name__)

TABLE_SESSIONS = "XXGSC_KM_SESSIONS"
TABLE_MESSAGES = "XXGSC_KM_MESSAGES"

ROLE_MAP_DB_TO_API = {
    "user": "user",
    "assistant": "assistant",
    "chatbot": "assistant",
    "bot": "assistant",
}

ROLE_MAP_API_TO_DB = {
    "user": "user",
    "assistant": "assistant",
}


def _normalize_role_from_db(role: Any) -> Optional[str]:
    normalized = str(role or "").strip().lower()
    return ROLE_MAP_DB_TO_API.get(normalized)


def _normalize_role_for_db(role: str) -> str:
    normalized = (role or "").strip().lower()
    return ROLE_MAP_API_TO_DB.get(normalized, "assistant")


def _read_lob_if_needed(value: Any) -> Any:
    return value.read() if hasattr(value, "read") else value


def ensure_session(session_id: str, title: Optional[str] = None, user_name: Optional[str] = None) -> None:
    if not session_id:
        return

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            MERGE INTO {TABLE_SESSIONS} tgt
            USING (
                SELECT HEXTORAW(:session_id) AS session_id,
                       :user_name AS user_name,
                       :title AS session_title
                FROM dual
            ) src
            ON (tgt.SESSION_ID = src.session_id)
            WHEN NOT MATCHED THEN INSERT (
                SESSION_ID,
                USER_NAME,
                SESSION_TITLE,
                START_AT,
                STATUS,
                LAST_MESSAGE_AT,
                CREATED_AT,
                UPDATED_AT
            ) VALUES (
                src.session_id,
                src.user_name,
                src.session_title,
                SYSTIMESTAMP,
                'ACTIVE',
                SYSTIMESTAMP,
                SYSTIMESTAMP,
                SYSTIMESTAMP
            )
            WHEN MATCHED THEN UPDATE SET
                tgt.UPDATED_AT = SYSTIMESTAMP,
                tgt.LAST_MESSAGE_AT = SYSTIMESTAMP,
                tgt.USER_NAME = COALESCE(tgt.USER_NAME, src.user_name)
            """,
            {
                "session_id": session_id,
                "user_name": user_name,
                "title": title,
            },
        )
        conn.commit()
    finally:
        cur.close()
        close_connection(conn)


def set_session_title_if_empty(session_id: str, title: str) -> None:
    if not session_id or not (title or "").strip():
        return

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            UPDATE {TABLE_SESSIONS}
            SET SESSION_TITLE = :title,
                UPDATED_AT = SYSTIMESTAMP
            WHERE SESSION_ID = HEXTORAW(:session_id)
              AND (SESSION_TITLE IS NULL OR TRIM(SESSION_TITLE) = '')
            """,
            {
                "session_id": session_id,
                "title": title,
            },
        )
        conn.commit()
    finally:
        cur.close()
        close_connection(conn)


def touch_session(session_id: str) -> None:
    if not session_id:
        return

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            UPDATE {TABLE_SESSIONS}
            SET UPDATED_AT = SYSTIMESTAMP,
                LAST_MESSAGE_AT = SYSTIMESTAMP,
                STATUS = 'ACTIVE'
            WHERE SESSION_ID = HEXTORAW(:session_id)
            """,
            {"session_id": session_id},
        )
        conn.commit()
    finally:
        cur.close()
        close_connection(conn)


def insert_session_message(
    session_id: str,
    role: str,
    content: str,
    *,
    sender_name: Optional[str] = None,
    model_name: Optional[str] = None,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    latency_ms: Optional[int] = None,
    metadata_json: Optional[str] = None,
) -> None:
    if not session_id or not (content or "").strip():
        return

    ensure_session(session_id)
    db_role = _normalize_role_for_db(role)
    is_own_yn = "Y" if db_role == "user" else "N"
    resolved_sender = sender_name or ("Assistant" if db_role == "assistant" else None)

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.setinputsizes(
            session_id=oracledb.DB_TYPE_VARCHAR,
            message_role=oracledb.DB_TYPE_VARCHAR,
            sender_name=oracledb.DB_TYPE_VARCHAR,
            is_own_yn=oracledb.DB_TYPE_VARCHAR,
            content_text=oracledb.DB_TYPE_VARCHAR,
            model_name=oracledb.DB_TYPE_VARCHAR,
            metadata_json=oracledb.DB_TYPE_VARCHAR,
        )
        cur.execute(
            f"""
            INSERT INTO {TABLE_MESSAGES} (
                MESSAGE_ID,
                SESSION_ID,
                MESSAGE_ROLE,
                SENDER_NAME,
                IS_OWN_YN,
                CONTENT_TEXT,
                CREATED_AT,
                MODEL_NAME,
                TOKENS_IN,
                TOKENS_OUT,
                LATENCY_MS,
                METADATA_JSON
            ) VALUES (
                SYS_GUID(),
                HEXTORAW(:session_id),
                :message_role,
                :sender_name,
                :is_own_yn,
                :content_text,
                SYSTIMESTAMP,
                :model_name,
                :tokens_in,
                :tokens_out,
                :latency_ms,
                :metadata_json
            )
            """,
            {
                "session_id": session_id,
                "message_role": db_role,
                "sender_name": resolved_sender,
                "is_own_yn": is_own_yn,
                "content_text": content,
                "model_name": model_name,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "metadata_json": metadata_json,
            },
        )
        conn.commit()
    finally:
        cur.close()
        close_connection(conn)

    touch_session(session_id)


def fetch_session_history(session_id: str, limit: int = 20) -> List[Dict[str, str]]:
    if not session_id:
        return []

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT MESSAGE_ROLE, CONTENT_TEXT
            FROM (
                SELECT MESSAGE_ID, MESSAGE_ROLE, CONTENT_TEXT, CREATED_AT
                FROM {TABLE_MESSAGES}
                WHERE SESSION_ID = HEXTORAW(:session_id)
                ORDER BY CREATED_AT DESC, MESSAGE_ID DESC
            )
            WHERE ROWNUM <= :limit
            ORDER BY CREATED_AT ASC, MESSAGE_ID ASC
            """,
            {
                "session_id": session_id,
                "limit": limit,
            },
        )

        history: List[Dict[str, str]] = []
        for db_role, content in cur:
            normalized_role = _normalize_role_from_db(db_role)
            if not normalized_role:
                continue
            content = _read_lob_if_needed(content)
            text = str(content or "").strip()
            if not text:
                continue
            history.append({"role": normalized_role, "content": text})
        return history
    finally:
        cur.close()
        close_connection(conn)


def delete_session_history(session_id: str) -> int:
    if not session_id:
        return 0

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            DELETE FROM {TABLE_MESSAGES}
            WHERE SESSION_ID = HEXTORAW(:session_id)
            """,
            {"session_id": session_id},
        )
        deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        cur.close()
        close_connection(conn)


def delete_session(session_id: str) -> Dict[str, int]:
    if not session_id:
        return {"history_deleted": 0, "session_deleted": 0}

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            DELETE FROM {TABLE_MESSAGES}
            WHERE SESSION_ID = HEXTORAW(:session_id)
            """,
            {"session_id": session_id},
        )
        history_deleted = cur.rowcount

        cur.execute(
            f"""
            DELETE FROM {TABLE_SESSIONS}
            WHERE SESSION_ID = HEXTORAW(:session_id)
            """,
            {"session_id": session_id},
        )
        session_deleted = cur.rowcount
        conn.commit()
        return {
            "history_deleted": history_deleted,
            "session_deleted": session_deleted,
        }
    finally:
        cur.close()
        close_connection(conn)


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    if not session_id:
        return None

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT RAWTOHEX(SESSION_ID), USER_NAME, SESSION_TITLE, START_AT, END_AT, STATUS, LAST_MESSAGE_AT, CREATED_AT, UPDATED_AT
            FROM {TABLE_SESSIONS}
            WHERE SESSION_ID = HEXTORAW(:session_id)
            """,
            {"session_id": session_id},
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "session_id": row[0],
            "user_name": row[1],
            "title": row[2],
            "start_at": row[3].isoformat() if row[3] else None,
            "end_at": row[4].isoformat() if row[4] else None,
            "status": row[5],
            "last_message_at": row[6].isoformat() if row[6] else None,
            "created_at": row[7].isoformat() if row[7] else None,
            "updated_at": row[8].isoformat() if row[8] else None,
        }
    finally:
        cur.close()
        close_connection(conn)


def list_sessions(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT RAWTOHEX(SESSION_ID), USER_NAME, SESSION_TITLE, START_AT, END_AT, STATUS, LAST_MESSAGE_AT, CREATED_AT, UPDATED_AT
            FROM {TABLE_SESSIONS}
            ORDER BY NVL(LAST_MESSAGE_AT, CREATED_AT) DESC
            OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
            """,
            {"offset": offset, "limit": limit},
        )
        rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "session_id": row[0],
                    "user_name": row[1],
                    "title": row[2],
                    "start_at": row[3].isoformat() if row[3] else None,
                    "end_at": row[4].isoformat() if row[4] else None,
                    "status": row[5],
                    "last_message_at": row[6].isoformat() if row[6] else None,
                    "created_at": row[7].isoformat() if row[7] else None,
                    "updated_at": row[8].isoformat() if row[8] else None,
                }
            )
        return results
    finally:
        cur.close()
        close_connection(conn)