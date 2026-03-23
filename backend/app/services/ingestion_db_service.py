import hashlib
import logging
from typing import Any, Dict, Optional

from app.services.vector_store_service import close_connection, get_connection

logger = logging.getLogger(__name__)

TABLE_PROJECTS = "XXGSC_KM_PROJECTS"
TABLE_BATCH = "XXGSC_KM_INGESTION_BATCH"
TABLE_RUN = "XXGSC_KM_INGESTION_RUN"
TABLE_DOCUMENTS = "XXGSC_KM_DOCUMENTS"
TABLE_DOCUMENT_VERSION = "XXGSC_KM_DOCUMENT_VERSION"
TABLE_STEP_LOG = "XXGSC_KM_INGESTION_STEP_LOG"
TABLE_CHUNK = "XXGSC_KM_DOCUMENT_CHUNK"
TABLE_VECTOR = "XXGSC_KM_CHUNK_VECTOR"


def _lob_to_str(value: Any) -> Any:
    return value.read() if hasattr(value, "read") else value


def _release(conn) -> None:
    close_connection(conn)


def get_or_create_project(project: Dict[str, Any], created_by: str) -> Dict[str, str]:
    project_name = (project.get("project_name") or "DEFAULT_PROJECT").strip()
    if not project_name:
        raise ValueError("project_name is required")

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            f"SELECT PROJECT_ID FROM {TABLE_PROJECTS} WHERE PROJECT_NAME = :project_name FETCH FIRST 1 ROWS ONLY",
            {"project_name": project_name},
        )
        row = cur.fetchone()
        if row:
            return {"project_id": row[0], "project_name": project_name}

        cur.execute(
            f"""
            INSERT INTO {TABLE_PROJECTS} (
                PROJECT_ID,
                PROJECT_NAME,
                GEOGRAPHY_CODE,
                VERTICAL_CODE,
                ENGAGEMENT_TYPE,
                CONFIDENTIALITY,
                CREATED_BY,
                CREATED_DATE,
                LAST_UPDATED_BY,
                LAST_UPDATED_DATE
            ) VALUES (
                RAWTOHEX(SYS_GUID()),
                :project_name,
                :geography_code,
                :vertical_code,
                :engagement_type,
                :confidentiality,
                :created_by,
                CURRENT_TIMESTAMP,
                :created_by,
                CURRENT_TIMESTAMP
            )
            RETURNING PROJECT_ID INTO :project_id
            """,
            {
                "project_name": project_name,
                "geography_code": project.get("geography_code"),
                "vertical_code": project.get("vertical_code"),
                "engagement_type": project.get("engagement_type"),
                "confidentiality": project.get("confidentiality"),
                "created_by": created_by,
                "project_id": cur.var(str),
            },
        )
        project_id = cur.getimplicitresults() if False else None
        conn.commit()

        cur.execute(
            f"SELECT PROJECT_ID FROM {TABLE_PROJECTS} WHERE PROJECT_NAME = :project_name FETCH FIRST 1 ROWS ONLY",
            {"project_name": project_name},
        )
        inserted = cur.fetchone()
        return {"project_id": inserted[0], "project_name": project_name}
    finally:
        cur.close()
        _release(conn)


def create_ingestion_batch(source_system: str, requested_by: str, total_documents: int) -> str:
    conn = get_connection()
    cur = conn.cursor()

    try:
        batch_id_var = cur.var(str)
        cur.execute(
            f"""
            INSERT INTO {TABLE_BATCH} (
                BATCH_ID,
                SOURCE_SYSTEM,
                REQUESTED_BY,
                REQUESTED_AT,
                STATUS,
                TOTAL_DOCUMENTS,
                SUCCESSFUL_DOCUMENTS,
                FAILED_DOCUMENTS,
                CREATED_BY,
                CREATION_DATE,
                LAST_UPDATED_BY,
                LAST_UPDATE_DATE
            ) VALUES (
                RAWTOHEX(SYS_GUID()),
                :source_system,
                :requested_by,
                CURRENT_TIMESTAMP,
                'IN_PROGRESS',
                :total_documents,
                0,
                0,
                :requested_by,
                CURRENT_TIMESTAMP,
                :requested_by,
                CURRENT_TIMESTAMP
            )
            RETURNING BATCH_ID INTO :batch_id
            """,
            {
                "source_system": source_system,
                "requested_by": requested_by,
                "total_documents": total_documents,
                "batch_id": batch_id_var,
            },
        )
        conn.commit()
        return batch_id_var.getvalue()[0]
    finally:
        cur.close()
        _release(conn)


def create_ingestion_run(batch_id: str, triggered_by: str, total_documents: int) -> str:
    conn = get_connection()
    cur = conn.cursor()

    try:
        run_id_var = cur.var(str)
        cur.execute(
            f"""
            INSERT INTO {TABLE_RUN} (
                RUN_ID,
                BATCH_ID,
                RUN_TYPE,
                STATUS,
                TRIGGERED_BY,
                STARTED_AT,
                TOTAL_DOCUMENTS,
                SUCCESSFUL_DOCUMENTS,
                FAILED_DOCUMENTS,
                CREATED_BY,
                CREATION_DATE,
                LAST_UPDATED_BY,
                LAST_UPDATE_DATE
            ) VALUES (
                RAWTOHEX(SYS_GUID()),
                :batch_id,
                'AUTO',
                'IN_PROGRESS',
                :triggered_by,
                CURRENT_TIMESTAMP,
                :total_documents,
                0,
                0,
                :triggered_by,
                CURRENT_TIMESTAMP,
                :triggered_by,
                CURRENT_TIMESTAMP
            )
            RETURNING RUN_ID INTO :run_id
            """,
            {
                "batch_id": batch_id,
                "triggered_by": triggered_by,
                "total_documents": total_documents,
                "run_id": run_id_var,
            },
        )
        conn.commit()
        return run_id_var.getvalue()[0]
    finally:
        cur.close()
        _release(conn)


def create_document_record(
    project_id: str,
    batch_id: str,
    file_name: str,
    requested_by: str,
    *,
    object_name: Optional[str] = None,
    bucket_name: Optional[str] = None,
    namespace_name: Optional[str] = None,
    object_uri: Optional[str] = None,
    module_code: Optional[str] = None,
    doc_type_code: Optional[str] = None,
    mime_type: Optional[str] = None,
    file_hash: Optional[str] = None,
    content_hash: Optional[str] = None,
    status: str = "IN_PROGRESS",
) -> str:
    conn = get_connection()
    cur = conn.cursor()

    try:
        document_id_var = cur.var(str)
        cur.execute(
            f"""
            INSERT INTO {TABLE_DOCUMENTS} (
                DOCUMENT_ID,
                PROJECT_ID,
                FILE_NAME,
                DOC_TYPE_CODE,
                MODULE_CODE,
                STATUS,
                RAG_COMPLIANT_FLAG,
                MIME_TYPE,
                CREATED_BY,
                CREATED_DATE,
                LAST_UPDATED_BY,
                LAST_UPDATED_DATE,
                ATTRIBUTE1,
                ATTRIBUTE2,
                ORIGINAL_FILE_NAME,
                OBJECT_NAME,
                BUCKET_NAME,
                NAMESPACE_NAME,
                OBJECT_URI,
                INGESTION_BATCH_ID
            ) VALUES (
                RAWTOHEX(SYS_GUID()),
                :project_id,
                :file_name,
                :doc_type_code,
                :module_code,
                :status,
                'Y',
                :mime_type,
                :requested_by,
                CURRENT_TIMESTAMP,
                :requested_by,
                CURRENT_TIMESTAMP,
                :file_hash,
                :content_hash,
                :file_name,
                :object_name,
                :bucket_name,
                :namespace_name,
                :object_uri,
                :batch_id
            )
            RETURNING DOCUMENT_ID INTO :document_id
            """,
            {
                "project_id": project_id,
                "file_name": file_name,
                "doc_type_code": doc_type_code,
                "module_code": module_code,
                "mime_type": mime_type,
                "requested_by": requested_by,
                "status": status,
                "file_hash": file_hash,
                "content_hash": content_hash,
                "object_name": object_name,
                "bucket_name": bucket_name,
                "namespace_name": namespace_name,
                "object_uri": object_uri,
                "batch_id": batch_id,
                "document_id": document_id_var,
            },
        )
        conn.commit()
        return document_id_var.getvalue()[0]
    finally:
        cur.close()
        _release(conn)


def create_document_version(document_id: str, object_name: str, created_by: str, version_no: int) -> str:
    conn = get_connection()
    cur = conn.cursor()
    try:
        version_id_var = cur.var(str)
        cur.execute(
            f"""
            INSERT INTO {TABLE_DOCUMENT_VERSION} (
                DOCUMENT_VERSION_ID,
                DOCUMENT_ID,
                VERSION_NO,
                OBJECT_NAME,
                IS_CURRENT,
                VALID_FROM,
                CREATED_BY,
                CREATION_DATE,
                LAST_UPDATED_BY,
                LAST_UPDATE_DATE
            ) VALUES (
                RAWTOHEX(SYS_GUID()),
                :document_id,
                :version_no,
                :object_name,
                'Y',
                CURRENT_TIMESTAMP,
                :created_by,
                CURRENT_TIMESTAMP,
                :created_by,
                CURRENT_TIMESTAMP
            )
            RETURNING DOCUMENT_VERSION_ID INTO :version_id
            """,
            {
                "document_id": document_id,
                "version_no": version_no,
                "object_name": object_name,
                "created_by": created_by,
                "version_id": version_id_var,
            },
        )
        conn.commit()
        return version_id_var.getvalue()[0]
    finally:
        cur.close()
        _release(conn)


def find_current_document_version(project_id: str, object_name: str, file_name: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT
                d.DOCUMENT_ID,
                d.FILE_NAME,
                d.OBJECT_NAME,
                d.ATTRIBUTE1,
                d.ATTRIBUTE2,
                dv.VERSION_NO
            FROM {TABLE_DOCUMENTS} d
            LEFT JOIN {TABLE_DOCUMENT_VERSION} dv
              ON dv.DOCUMENT_ID = d.DOCUMENT_ID
             AND dv.IS_CURRENT = 'Y'
            WHERE d.PROJECT_ID = :project_id
              AND d.STATUS != 'SUPERSEDED'
              AND (
                    d.OBJECT_NAME = :object_name
                 OR d.FILE_NAME = :file_name
              )
            ORDER BY NVL(dv.VERSION_NO, 0) DESC, d.CREATED_DATE DESC
            FETCH FIRST 1 ROWS ONLY
            """,
            {
                "project_id": project_id,
                "object_name": object_name,
                "file_name": file_name,
            },
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "document_id": row[0],
            "file_name": row[1],
            "object_name": row[2],
            "file_hash": row[3],
            "content_hash": row[4],
            "version_no": row[5] or 1,
        }
    finally:
        cur.close()
        _release(conn)


def supersede_document(document_id: str, updated_by: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            UPDATE {TABLE_DOCUMENTS}
            SET STATUS = 'SUPERSEDED',
                LAST_UPDATED_BY = :updated_by,
                LAST_UPDATED_DATE = CURRENT_TIMESTAMP
            WHERE DOCUMENT_ID = :document_id
            """,
            {"updated_by": updated_by, "document_id": document_id},
        )
        cur.execute(
            f"""
            UPDATE {TABLE_CHUNK}
            SET CHUNK_STATUS = 'SUPERSEDED',
                LAST_UPDATED_BY = :updated_by,
                LAST_UPDATE_DATE = CURRENT_TIMESTAMP
            WHERE DOCUMENT_ID = :document_id
            """,
            {"updated_by": updated_by, "document_id": document_id},
        )
        cur.execute(
            f"""
            UPDATE {TABLE_DOCUMENT_VERSION}
            SET IS_CURRENT = 'N',
                VALID_TO = CURRENT_TIMESTAMP,
                LAST_UPDATED_BY = :updated_by,
                LAST_UPDATE_DATE = CURRENT_TIMESTAMP
            WHERE DOCUMENT_ID = :document_id
              AND IS_CURRENT = 'Y'
            """,
            {"updated_by": updated_by, "document_id": document_id},
        )
        conn.commit()
    finally:
        cur.close()
        _release(conn)


def mark_document_status(document_id: str, status: str, updated_by: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            UPDATE {TABLE_DOCUMENTS}
            SET STATUS = :status,
                LAST_UPDATED_BY = :updated_by,
                LAST_UPDATED_DATE = CURRENT_TIMESTAMP
            WHERE DOCUMENT_ID = :document_id
            """,
            {"status": status, "updated_by": updated_by, "document_id": document_id},
        )
        conn.commit()
    finally:
        cur.close()
        _release(conn)


def finalize_ingestion_run(run_id: str, successful_documents: int, failed_documents: int, updated_by: str) -> None:
    status = "COMPLETED" if failed_documents == 0 else "PARTIAL_SUCCESS"
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            UPDATE {TABLE_RUN}
            SET STATUS = :status,
                ENDED_AT = CURRENT_TIMESTAMP,
                SUCCESSFUL_DOCUMENTS = :successful_documents,
                FAILED_DOCUMENTS = :failed_documents,
                LAST_UPDATED_BY = :updated_by,
                LAST_UPDATE_DATE = CURRENT_TIMESTAMP
            WHERE RUN_ID = :run_id
            """,
            {
                "status": status,
                "successful_documents": successful_documents,
                "failed_documents": failed_documents,
                "updated_by": updated_by,
                "run_id": run_id,
            },
        )
        conn.commit()
    finally:
        cur.close()
        _release(conn)


def finalize_ingestion_batch(batch_id: str, successful_documents: int, failed_documents: int, updated_by: str) -> None:
    status = "COMPLETED" if failed_documents == 0 else "PARTIAL_SUCCESS"
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            UPDATE {TABLE_BATCH}
            SET STATUS = :status,
                SUCCESSFUL_DOCUMENTS = :successful_documents,
                FAILED_DOCUMENTS = :failed_documents,
                LAST_UPDATED_BY = :updated_by,
                LAST_UPDATE_DATE = CURRENT_TIMESTAMP
            WHERE BATCH_ID = :batch_id
            """,
            {
                "status": status,
                "successful_documents": successful_documents,
                "failed_documents": failed_documents,
                "updated_by": updated_by,
                "batch_id": batch_id,
            },
        )
        conn.commit()
    finally:
        cur.close()
        _release(conn)


def get_batch_status(batch_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT
                BATCH_ID,
                STATUS,
                TOTAL_DOCUMENTS,
                SUCCESSFUL_DOCUMENTS,
                FAILED_DOCUMENTS,
                REQUESTED_BY,
                REQUESTED_AT,
                SOURCE_SYSTEM
            FROM {TABLE_BATCH}
            WHERE BATCH_ID = :batch_id
            """,
            {"batch_id": batch_id},
        )
        row = cur.fetchone()
        if not row:
            return None
        total_documents = row[2] or 0
        successful_documents = row[3] or 0
        failed_documents = row[4] or 0
        return {
            "batch_id": row[0],
            "status": row[1],
            "total_documents": total_documents,
            "successful_documents": successful_documents,
            "failed_documents": failed_documents,
            "remaining_documents": max(total_documents - successful_documents - failed_documents, 0),
            "requested_by": row[5],
            "requested_at": str(row[6]) if row[6] else None,
            "source_system": row[7],
        }
    finally:
        cur.close()
        _release(conn)


def get_documents_for_batch(batch_id: str) -> list[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT
                DOCUMENT_ID,
                PROJECT_ID,
                FILE_NAME,
                STATUS,
                DOC_TYPE_CODE,
                MODULE_CODE,
                MIME_TYPE,
                OBJECT_NAME,
                BUCKET_NAME,
                NAMESPACE_NAME
            FROM {TABLE_DOCUMENTS}
            WHERE INGESTION_BATCH_ID = :batch_id
            ORDER BY CREATED_DATE DESC
            """,
            {"batch_id": batch_id},
        )
        rows = cur.fetchall()
        return [
            {
                "document_id": row[0],
                "project_id": row[1],
                "file_name": row[2],
                "status": row[3],
                "doc_type_code": row[4],
                "module_code": row[5],
                "mime_type": row[6],
                "object_name": row[7],
                "bucket_name": row[8],
                "namespace_name": row[9],
            }
            for row in rows
        ]
    finally:
        cur.close()
        _release(conn)


def get_documents_for_batch_by_ids(batch_id: str, document_ids: list[str]) -> list[Dict[str, Any]]:
    if not document_ids:
        return []

    conn = get_connection()
    cur = conn.cursor()
    try:
        bind_names = []
        binds: Dict[str, Any] = {"batch_id": batch_id}
        for idx, document_id in enumerate(document_ids):
            key = f"doc_id_{idx}"
            bind_names.append(f":{key}")
            binds[key] = document_id

        cur.execute(
            f"""
            SELECT
                DOCUMENT_ID,
                PROJECT_ID,
                FILE_NAME,
                STATUS,
                DOC_TYPE_CODE,
                MODULE_CODE,
                MIME_TYPE,
                OBJECT_NAME,
                BUCKET_NAME,
                NAMESPACE_NAME
            FROM {TABLE_DOCUMENTS}
            WHERE INGESTION_BATCH_ID = :batch_id
              AND DOCUMENT_ID IN ({', '.join(bind_names)})
            ORDER BY CREATED_DATE DESC
            """,
            binds,
        )
        rows = cur.fetchall()
        return [
            {
                "document_id": row[0],
                "project_id": row[1],
                "file_name": row[2],
                "status": row[3],
                "doc_type_code": row[4],
                "module_code": row[5],
                "mime_type": row[6],
                "object_name": row[7],
                "bucket_name": row[8],
                "namespace_name": row[9],
            }
            for row in rows
        ]
    finally:
        cur.close()
        _release(conn)


def list_documents_by_batch(batch_id: str) -> list[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT
                DOCUMENT_ID,
                FILE_NAME,
                STATUS,
                DOC_TYPE_CODE,
                MODULE_CODE,
                CREATED_DATE,
                LAST_UPDATED_DATE
            FROM {TABLE_DOCUMENTS}
            WHERE INGESTION_BATCH_ID = :batch_id
            ORDER BY CREATED_DATE DESC
            """,
            {"batch_id": batch_id},
        )
        rows = cur.fetchall()
        return [
            {
                "document_id": row[0],
                "file_name": row[1],
                "status": row[2],
                "doc_type_code": row[3],
                "module_code": row[4],
                "created_date": str(row[5]) if row[5] else None,
                "last_updated_date": str(row[6]) if row[6] else None,
            }
            for row in rows
        ]
    finally:
        cur.close()
        _release(conn)


def list_document_steps(document_id: str) -> list[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT
                LOG_ID,
                STEP_NAME,
                STEP_STATUS,
                STEP_SEQUENCE,
                MESSAGE,
                STARTED_AT,
                ENDED_AT,
                DURATION_MS
            FROM {TABLE_STEP_LOG}
            WHERE DOCUMENT_ID = :document_id
            ORDER BY STEP_SEQUENCE, STARTED_AT
            """,
            {"document_id": document_id},
        )
        rows = cur.fetchall()
        return [
            {
                "log_id": row[0],
                "step_name": row[1],
                "step_status": row[2],
                "step_sequence": row[3],
                "message": _lob_to_str(row[4]),
                "started_at": str(row[5]) if row[5] else None,
                "ended_at": str(row[6]) if row[6] else None,
                "duration_ms": row[7],
            }
            for row in rows
        ]
    finally:
        cur.close()
        _release(conn)


def start_step_log(
    run_id: str,
    batch_id: str,
    document_id: str,
    step_name: str,
    step_sequence: int,
    created_by: str,
) -> str:
    conn = get_connection()
    cur = conn.cursor()
    try:
        log_id_var = cur.var(str)
        cur.execute(
            f"""
            INSERT INTO {TABLE_STEP_LOG} (
                LOG_ID,
                RUN_ID,
                BATCH_ID,
                DOCUMENT_ID,
                STEP_NAME,
                STEP_STATUS,
                STEP_SEQUENCE,
                MESSAGE,
                STARTED_AT,
                CREATED_BY,
                CREATION_DATE,
                LAST_UPDATED_BY,
                LAST_UPDATE_DATE
            ) VALUES (
                RAWTOHEX(SYS_GUID()),
                :run_id,
                :batch_id,
                :document_id,
                :step_name,
                'STARTED',
                :step_sequence,
                :message,
                CURRENT_TIMESTAMP,
                :created_by,
                CURRENT_TIMESTAMP,
                :created_by,
                CURRENT_TIMESTAMP
            )
            RETURNING LOG_ID INTO :log_id
            """,
            {
                "run_id": run_id,
                "batch_id": batch_id,
                "document_id": document_id,
                "step_name": step_name,
                "step_sequence": step_sequence,
                "message": f"{step_name} started",
                "created_by": created_by,
                "log_id": log_id_var,
            },
        )
        conn.commit()
        return log_id_var.getvalue()[0]
    finally:
        cur.close()
        _release(conn)


def finish_step_log(log_id: str, status: str, message: str, updated_by: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            UPDATE {TABLE_STEP_LOG}
            SET STEP_STATUS = :status,
                MESSAGE = :message,
                ENDED_AT = CURRENT_TIMESTAMP,
                DURATION_MS = ROUND((CAST(CURRENT_TIMESTAMP AS DATE) - CAST(STARTED_AT AS DATE)) * 86400000),
                LAST_UPDATED_BY = :updated_by,
                LAST_UPDATE_DATE = CURRENT_TIMESTAMP
            WHERE LOG_ID = :log_id
            """,
            {
                "status": status,
                "message": message,
                "updated_by": updated_by,
                "log_id": log_id,
            },
        )
        conn.commit()
    finally:
        cur.close()
        _release(conn)


def delete_document_completely(document_id: str) -> Dict[str, int]:
    conn = get_connection()
    cur = conn.cursor()
    counts: Dict[str, int] = {}
    try:
        cur.execute(f"DELETE FROM {TABLE_VECTOR} WHERE DOCUMENT_ID = :document_id", {"document_id": document_id})
        counts["chunk_vectors"] = cur.rowcount
        cur.execute(f"DELETE FROM {TABLE_CHUNK} WHERE DOCUMENT_ID = :document_id", {"document_id": document_id})
        counts["document_chunks"] = cur.rowcount
        cur.execute(f"DELETE FROM {TABLE_STEP_LOG} WHERE DOCUMENT_ID = :document_id", {"document_id": document_id})
        counts["step_logs"] = cur.rowcount
        cur.execute(f"DELETE FROM {TABLE_DOCUMENT_VERSION} WHERE DOCUMENT_ID = :document_id", {"document_id": document_id})
        counts["document_versions"] = cur.rowcount
        cur.execute(f"DELETE FROM {TABLE_DOCUMENTS} WHERE DOCUMENT_ID = :document_id", {"document_id": document_id})
        counts["documents"] = cur.rowcount
        conn.commit()
        return counts
    finally:
        cur.close()
        _release(conn)


def compute_object_uri(namespace_name: Optional[str], bucket_name: Optional[str], object_name: Optional[str]) -> Optional[str]:
    if not namespace_name or not bucket_name or not object_name:
        return None
    return f"oci://{bucket_name}@{namespace_name}/{object_name}"


def make_deterministic_chunk_id(document_id: str, chunk_index: int, chunk_text: str) -> str:
    seed = f"{document_id}|{chunk_index}|{hashlib.sha256((chunk_text or '').encode('utf-8')).hexdigest()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def compute_sha256_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()