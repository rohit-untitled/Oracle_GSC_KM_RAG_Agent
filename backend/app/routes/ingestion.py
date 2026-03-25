import hashlib
import logging
import mimetypes
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.services.anonymize_service import anonymize_markdown
from app.services.chunk_service import _chunk_blocks, _merge_consecutive_tables, _parse_markdown_blocks
from app.services.extractors.document_extractor_service import extract_text_with_formatting_in_sequence
from app.services.document_loader import load_docx_files
from app.services.embedding_service import OCIEmbeddingService
from app.services.ingestion_db_service import (
    claim_ready_document,
    compute_sha256_text,
    create_document_version,
    create_ingestion_run,
    delete_document_completely,
    finalize_ingestion_batch,
    finalize_ingestion_run,
    find_current_document_version,
    finish_step_log,
    get_batch_status,
    get_document_created_by,
    get_current_document_version,
    get_documents_for_batch,
    get_documents_for_batch_by_ids,
    get_ready_documents,
    list_documents_by_batch,
    list_document_steps,
    make_deterministic_chunk_id,
    mark_document_status,
    start_step_log,
    supersede_document,
)
from app.services.oci_downloader import download_all_from_bucket, download_object
from app.services.secure_config import get_env
from app.services.vector_store_service import insert_document_embeddings
from app.services.vector_store_service import get_connection, close_connection


router = APIRouter()
logger = logging.getLogger("KM Knowledge Agent is Working")

DEFAULT_BUCKET = get_env("BUCKET_NAME")
DEFAULT_NAMESPACE = get_env("OCI_NAMESPACE")
AUTO_INGEST_ENABLED = (get_env("AUTO_INGEST_ENABLED", "true") or "true").strip().lower() in {"1", "true", "yes", "y"}
AUTO_INGEST_POLL_SECONDS = int(get_env("AUTO_INGEST_POLL_SECONDS", "30"))
AUTO_INGEST_BATCH_SIZE = int(get_env("AUTO_INGEST_BATCH_SIZE", "10"))
DEFAULT_CHUNK_MAX_TOKENS = 300
DEFAULT_CHUNK_OVERLAP_TOKENS = 40


class SelectedDocumentPayload(BaseModel):
    document_id: str


class IngestionApiRequest(BaseModel):
    batch_id: str = Field(..., description="Existing batch identifier created by APEX")
    requested_by: str = Field(..., description="User or system triggering ingestion")
    source_system: str = Field(default="APEX", description="Source system name")
    anonymize_docs: bool = True
    documents: List[SelectedDocumentPayload] = Field(default_factory=list)


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_docs_folder() -> str:
    return str(_backend_root() / "app" / "data" / "downloads")


def _extract_document_text(local_path: str) -> str:
    return extract_text_with_formatting_in_sequence(local_path)


def _compute_file_hash(local_path: str) -> str:
    with open(local_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _run_logged_step(
    *,
    run_id: str,
    batch_id: str,
    document_id: str,
    step_name: str,
    step_sequence: int,
    created_by: str,
    action,
):
    log_id = start_step_log(run_id, batch_id, document_id, step_name, step_sequence, created_by)
    try:
        result = action()
        finish_step_log(log_id, "COMPLETED", f"{step_name} completed", created_by)
        return result
    except Exception as exc:
        finish_step_log(log_id, "FAILED", str(exc), created_by)
        raise


def _chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> List[Dict[str, str]]:
    blocks = _parse_markdown_blocks(text)
    blocks = _merge_consecutive_tables(blocks)
    return _chunk_blocks(blocks, max_tokens=max_tokens, overlap_tokens=overlap_tokens)


def _embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    embedder = OCIEmbeddingService()
    texts = [c.get("chunk", "") for c in chunks]
    vectors = embedder.embed_texts(texts)

    output: List[Dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors):
        row = dict(chunk)
        row["embedding"] = vector
        output.append(row)
    return output


def _process_document(
    doc: Dict[str, Any],
    *,
    batch_id: str,
    run_id: str,
    project_id: str,
    requested_by: str,
    anonymize_docs: bool,
    chunk_max_tokens: int,
    chunk_overlap_tokens: int,
) -> Dict[str, Any]:
    document_id = doc.get("document_id")
    file_name = doc.get("file_name")
    object_name = doc.get("object_name")
    try:
        mark_document_status(document_id, "IN_PROGRESS", requested_by)

        def _prepare_document() -> Dict[str, Any]:
            if not document_id:
                raise ValueError("document_id is missing for ingestion document")
            if not file_name:
                raise ValueError(f"file_name is missing for document {document_id}")
            if not object_name:
                raise ValueError(f"OBJECT_NAME is missing in DB for document {document_id}")

            local_path = download_object(
                object_name,
                bucket_name=DEFAULT_BUCKET,
                namespace_name=DEFAULT_NAMESPACE,
            )
            resolved_mime_type = doc.get("mime_type") or mimetypes.guess_type(file_name)[0]
            file_hash = _compute_file_hash(local_path)
            extracted_text = _extract_document_text(local_path)
            content_hash = compute_sha256_text(extracted_text)

            existing = find_current_document_version(project_id, object_name, file_name) if project_id else None
            current_version = get_current_document_version(document_id)
            if existing and existing.get("document_id") != document_id:
                if existing.get("file_hash") == file_hash and existing.get("content_hash") == content_hash:
                    return {
                        "duplicate": True,
                        "mime_type": resolved_mime_type,
                        "extracted_text": extracted_text,
                    }

                supersede_document(existing["document_id"], requested_by)
                next_version = int(existing.get("version_no") or 1) + 1
            else:
                next_version = 1

            if current_version is None and (existing is None or existing.get("document_id") != document_id or next_version == 1):
                create_document_version(document_id, object_name, requested_by, next_version)

            return {
                "duplicate": False,
                "mime_type": resolved_mime_type,
                "extracted_text": extracted_text,
            }

        prepare_result = _run_logged_step(
            run_id=run_id,
            batch_id=batch_id,
            document_id=document_id,
            step_name="PREPARE",
            step_sequence=1,
            created_by=requested_by,
            action=_prepare_document,
        )

        if prepare_result.get("duplicate"):
            mark_document_status(document_id, "COMPLETED", requested_by)
            return {
                "document_id": document_id,
                "file_name": file_name,
                "status": "SKIPPED_DUPLICATE",
                "message": "Duplicate document detected by file/content hash",
            }

        extracted_text = prepare_result["extracted_text"]

        processed_text = _run_logged_step(
            run_id=run_id,
            batch_id=batch_id,
            document_id=document_id,
            step_name="ANONYMIZE",
            step_sequence=2,
            created_by=requested_by,
            action=lambda: anonymize_markdown(extracted_text) if anonymize_docs else extracted_text,
        )

        chunk_rows = _run_logged_step(
            run_id=run_id,
            batch_id=batch_id,
            document_id=document_id,
            step_name="CHUNK",
            step_sequence=3,
            created_by=requested_by,
            action=lambda: _chunk_text(processed_text, chunk_max_tokens, chunk_overlap_tokens),
        )
        prepared_chunks: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunk_rows):
            chunk_text = chunk.get("chunk", "")
            prepared_chunks.append(
                {
                    "chunk_id": make_deterministic_chunk_id(document_id, idx, chunk_text),
                    "chunk_index": idx,
                    "heading": chunk.get("heading"),
                    "source_file": file_name,
                    "chunk": chunk_text,
                }
            )

        embedded_chunks = _run_logged_step(
            run_id=run_id,
            batch_id=batch_id,
            document_id=document_id,
            step_name="EMBED",
            step_sequence=4,
            created_by=requested_by,
            action=lambda: _embed_chunks(prepared_chunks),
        )
        stored_chunks = _run_logged_step(
            run_id=run_id,
            batch_id=batch_id,
            document_id=document_id,
            step_name="STORE_VECTOR",
            step_sequence=5,
            created_by=requested_by,
            action=lambda: insert_document_embeddings(
                document_id=document_id,
                run_id=run_id,
                chunks=embedded_chunks,
                created_by=requested_by,
            ),
        )

        mark_document_status(document_id, "COMPLETED", requested_by)
        return {
            "document_id": document_id,
            "file_name": file_name,
            "status": "COMPLETED",
            "chunks_created": len(prepared_chunks),
            "vectors_stored": stored_chunks,
        }
    except Exception as exc:
        mark_document_status(document_id, "FAILED", requested_by)
        return {
            "document_id": document_id,
            "file_name": file_name,
            "status": "FAILED",
            "error": str(exc),
        }


@router.get("/sync-bucket")
def sync_bucket():
    try:
        download_all_from_bucket()
        return {"message": "All documents downloaded from OCI."}
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(500, str(e))


@router.get("/load-docs")
def load_docs():
    docs = load_docx_files(get_docs_folder())
    return {
        "total_documents": len(docs),
        "documents": [
            {
                "file": doc["file_name"],
                "folder": doc["folder"],
                "path": doc["file_path"],
            }
            for doc in docs
        ],
    }


@router.get("/db-test")
def db_test():
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT USER FROM dual")
        row = cur.fetchone()
        return {"status": "ok", "db_user": row[0] if row else None}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Database connectivity test failed.",
                "error": str(exc),
            },
        ) from exc
    finally:
        if cur:
            cur.close()
        close_connection(conn)


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    try:
        deleted = delete_document_completely(document_id)
        return {
            "status": "ok",
            "document_id": document_id,
            "deleted": deleted,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Document deletion failed.",
                "error": str(exc),
            },
        ) from exc


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str):
    batch = get_batch_status(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found.")
    return batch


@router.get("/batches/{batch_id}/documents")
def get_batch_documents(batch_id: str):
    batch = get_batch_status(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found.")
    return {
        "batch_id": batch_id,
        "documents": list_documents_by_batch(batch_id),
    }


@router.get("/documents/{document_id}/steps")
def get_document_steps(document_id: str):
    return {
        "document_id": document_id,
        "steps": list_document_steps(document_id),
    }


def _process_batch_documents(payload: IngestionApiRequest, batch_id: str, run_id: str) -> None:
    successful_documents = 0
    failed_documents = 0

    selected_document_ids = [doc.document_id for doc in payload.documents]
    if selected_document_ids:
        documents = get_documents_for_batch_by_ids(batch_id, selected_document_ids)
    else:
        documents = get_documents_for_batch(batch_id)

    for doc in documents:
        result = _process_document(
            doc,
            batch_id=batch_id,
            run_id=run_id,
            project_id=doc.get("project_id"),
            requested_by=payload.requested_by,
            anonymize_docs=payload.anonymize_docs,
            chunk_max_tokens=DEFAULT_CHUNK_MAX_TOKENS,
            chunk_overlap_tokens=DEFAULT_CHUNK_OVERLAP_TOKENS,
        )
        if result["status"] in {"COMPLETED", "SKIPPED_DUPLICATE"}:
            successful_documents += 1
        else:
            failed_documents += 1

    finalize_ingestion_run(run_id, successful_documents, failed_documents, payload.requested_by)
    finalize_ingestion_batch(batch_id, successful_documents, failed_documents, payload.requested_by)


def process_ready_documents_once(triggered_by: str = "AUTO_WORKER") -> Dict[str, int]:
    ready_docs = get_ready_documents(limit=AUTO_INGEST_BATCH_SIZE)
    claimed = 0
    processed = 0
    completed = 0
    failed = 0

    for doc in ready_docs:
        document_id = doc.get("document_id")
        batch_id = doc.get("batch_id")
        requested_by = doc.get("created_by") or doc.get("requested_by") or get_document_created_by(document_id) or triggered_by
        if not document_id or not batch_id:
            continue

        if not claim_ready_document(document_id, requested_by):
            continue

        claimed += 1
        run_id = create_ingestion_run(batch_id, requested_by, 1)
        result = _process_document(
            doc,
            batch_id=batch_id,
            run_id=run_id,
            project_id=doc.get("project_id"),
            requested_by=requested_by,
            anonymize_docs=True,
            chunk_max_tokens=DEFAULT_CHUNK_MAX_TOKENS,
            chunk_overlap_tokens=DEFAULT_CHUNK_OVERLAP_TOKENS,
        )
        processed += 1

        if result.get("status") in {"COMPLETED", "SKIPPED_DUPLICATE"}:
            completed += 1
            finalize_ingestion_run(run_id, 1, 0, requested_by)
            finalize_ingestion_batch(batch_id, 1, 0, requested_by)
        else:
            failed += 1
            finalize_ingestion_run(run_id, 0, 1, requested_by)
            finalize_ingestion_batch(batch_id, 0, 1, requested_by)

    return {
        "ready_found": len(ready_docs),
        "claimed": claimed,
        "processed": processed,
        "completed": completed,
        "failed": failed,
    }


def start_auto_ingestion_worker() -> None:
    if not AUTO_INGEST_ENABLED:
        logger.info("Auto ingestion worker is disabled")
        return

    def _worker_loop() -> None:
        logger.info(
            "Auto ingestion poller started | poll_seconds=%s batch_size=%s",
            AUTO_INGEST_POLL_SECONDS,
            AUTO_INGEST_BATCH_SIZE,
        )
        while True:
            try:
                stats = process_ready_documents_once(triggered_by="AUTO_WORKER")
                if stats.get("processed"):
                    logger.info("Auto ingestion worker cycle: %s", stats)
            except Exception:
                logger.exception("Auto ingestion worker cycle failed")
            time.sleep(AUTO_INGEST_POLL_SECONDS)

    thread = threading.Thread(target=_worker_loop, name="auto-ingestion-worker", daemon=True)
    thread.start()


@router.post("/ingestion/start")
def run_ingestion(payload: IngestionApiRequest, background_tasks: BackgroundTasks):
    try:
        batch = get_batch_status(payload.batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found.")

        selected_document_ids = [doc.document_id for doc in payload.documents]
        if selected_document_ids:
            documents = get_documents_for_batch_by_ids(payload.batch_id, selected_document_ids)
            if len(documents) != len(set(selected_document_ids)):
                raise HTTPException(status_code=400, detail="One or more requested documents are invalid for the batch.")
        else:
            documents = get_documents_for_batch(payload.batch_id)

        if not documents:
            raise HTTPException(status_code=400, detail="No documents found for the batch.")

        run_id = create_ingestion_run(payload.batch_id, payload.requested_by, len(documents))

        background_tasks.add_task(
            _process_batch_documents,
            payload,
            payload.batch_id,
            run_id,
        )

        return {
            "status": "accepted",
            "message": "Ingestion request accepted. Processing started.",
            "batch_id": payload.batch_id,
            "run_id": run_id,
            "requested_by": payload.requested_by,
            "source_system": payload.source_system,
            "anonymize_docs": payload.anonymize_docs,
            "selected_documents": len(documents),
        }
    except Exception as exc:
        logger.exception("Ingestion pipeline failed")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Ingestion pipeline failed.",
                "error": str(exc),
            },
        ) from exc
