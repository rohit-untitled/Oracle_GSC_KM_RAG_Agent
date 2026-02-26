import json
import logging
from typing import Dict, Any, List, Optional

import oracledb
from app.services.vector_store_service import get_connection, get_pool

logger = logging.getLogger(__name__)

def ensure_session(session_id: str, title: Optional[str] = None) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            MERGE INTO rag_sessions rs
            USING (SELECT :session_id AS session_id, :title AS title FROM dual) src
            ON (rs.session_id = src.session_id)
            WHEN NOT MATCHED THEN
              INSERT (session_id, title, created_at, updated_at, last_message_at)
              VALUES (src.session_id, src.title, SYSTIMESTAMP, SYSTIMESTAMP, NULL)
            """,
            {
                "session_id": session_id,
                "title": title,
            },
        )
        conn.commit()
    finally:
        cur.close()
        get_pool().release(conn)


def set_session_title_if_empty(session_id: str, title: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            UPDATE rag_sessions
            SET title = :title,
                updated_at = SYSTIMESTAMP
            WHERE session_id = :session_id
              AND (title IS NULL OR TRIM(title) = '')
            """,
            {
                "session_id": session_id,
                "title": title,
            },
        )
        conn.commit()
    finally:
        cur.close()
        get_pool().release(conn)


def touch_session(session_id: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            UPDATE rag_sessions
            SET updated_at = SYSTIMESTAMP,
                last_message_at = SYSTIMESTAMP
            WHERE session_id = :session_id
            """,
            {"session_id": session_id},
        )
        conn.commit()
    finally:
        cur.close()
        get_pool().release(conn)


def insert_session_message(session_id: str, role: str, content: str) -> None:
    ensure_session(session_id)
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.setinputsizes(
            session_id=oracledb.DB_TYPE_VARCHAR,
            role=oracledb.DB_TYPE_VARCHAR,
            content=oracledb.DB_TYPE_CLOB,
        )
        cur.execute(
            """
            INSERT INTO rag_session_history (session_id, role, content, created_at)
            VALUES (:session_id, :role, :content, SYSTIMESTAMP)
            """,
            {
                "session_id": session_id,
                "role": role,
                "content": content,
            },
        )
        conn.commit()
    finally:
        cur.close()
        get_pool().release(conn)
    touch_session(session_id)


def fetch_session_history(session_id: str, limit: int = 20) -> List[Dict[str, str]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content, created_at
                FROM rag_session_history
                WHERE session_id = :session_id
                ORDER BY created_at DESC, id DESC
            )
            WHERE ROWNUM <= :limit
            ORDER BY created_at ASC, id ASC
            """,
            {
                "session_id": session_id,
                "limit": limit,
            },
        )

        history: List[Dict[str, str]] = []
        for role, content in cur:
            if hasattr(content, "read"):
                content = content.read()
            history.append({"role": role, "content": content})
        return history
    finally:
        cur.close()
        get_pool().release(conn)


def delete_session_history(session_id: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            DELETE FROM rag_session_history
            WHERE session_id = :session_id
            """,
            {"session_id": session_id},
        )
        deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        cur.close()
        get_pool().release(conn)


def delete_session(session_id: str) -> Dict[str, int]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            DELETE FROM rag_session_history
            WHERE session_id = :session_id
            """,
            {"session_id": session_id},
        )
        history_deleted = cur.rowcount
        cur.execute(
            """
            DELETE FROM rag_sessions
            WHERE session_id = :session_id
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
        get_pool().release(conn)


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT session_id, title, created_at, updated_at, last_message_at
            FROM rag_sessions
            WHERE session_id = :session_id
            """,
            {"session_id": session_id},
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "session_id": row[0],
            "title": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
            "updated_at": row[3].isoformat() if row[3] else None,
            "last_message_at": row[4].isoformat() if row[4] else None,
        }
    finally:
        cur.close()
        get_pool().release(conn)


def list_sessions(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT session_id, title, created_at, updated_at, last_message_at
            FROM rag_sessions
            ORDER BY NVL(last_message_at, created_at) DESC
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
                    "title": row[1],
                    "created_at": row[2].isoformat() if row[2] else None,
                    "updated_at": row[3].isoformat() if row[3] else None,
                    "last_message_at": row[4].isoformat() if row[4] else None,
                }
            )
        return results
    finally:
        cur.close()
        get_pool().release(conn)
