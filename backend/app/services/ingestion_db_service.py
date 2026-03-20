import hashlib
import logging
from typing import Any, Dict, Optional

from app.services.vector_store_service import get_connection, close_connection

logger = logging.getLogger(__name__)

TABLE_PROJECTS = "XXGSC_KM_PROJECTS"
TABLE_BATCH = "XXGSC_KM_INGESTION_BATCH"
TABLE_RUN = "XXGSC_KM_INGESTION_RUN"
TABLE_DOCUMENTS = "XXGSC_KM_DOCUMENTS"


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
                'IN_PROGRESS',
                'Y',
                :mime_type,
                :requested_by,
                CURRENT_TIMESTAMP,
                :requested_by,
                CURRENT_TIMESTAMP,
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


def compute_object_uri(namespace_name: Optional[str], bucket_name: Optional[str], object_name: Optional[str]) -> Optional[str]:
    if not namespace_name or not bucket_name or not object_name:
        return None
    return f"oci://{bucket_name}@{namespace_name}/{object_name}"


def make_deterministic_chunk_id(document_id: str, chunk_index: int, chunk_text: str) -> str:
    seed = f"{document_id}|{chunk_index}|{hashlib.sha256((chunk_text or '').encode('utf-8')).hexdigest()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()