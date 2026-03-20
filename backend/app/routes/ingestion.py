import logging
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.anonymize_service import anonymize_markdown
from app.services.chunk_service import _chunk_blocks, _merge_consecutive_tables, _parse_markdown_blocks
from app.services.docx_extractor import extract_text_with_formatting_in_sequence
from app.services.document_loader import load_docx_files
from app.services.embedding_service import OCIEmbeddingService
from app.services.ingestion_db_service import (
    compute_object_uri,
    create_document_record,
    create_ingestion_batch,
    create_ingestion_run,
    finalize_ingestion_batch,
    finalize_ingestion_run,
    get_or_create_project,
    make_deterministic_chunk_id,
    mark_document_status,
)
from app.services.oci_downloader import download_all_from_bucket
from app.services.secure_config import get_env
from app.services.vector_store_service import insert_document_embeddings
from app.services.vector_store_service import get_connection, close_connection


router = APIRouter()
logger = logging.getLogger("KM Knowledge Agent is Working")

DEFAULT_BUCKET = get_env("BUCKET_NAME")
DEFAULT_NAMESPACE = get_env("OCI_NAMESPACE")


class ProjectPayload(BaseModel):
    project_name: str
    geography_code: Optional[str] = None
    vertical_code: Optional[str] = None
    engagement_type: Optional[str] = None
    confidentiality: Optional[str] = None


class DocumentPayload(BaseModel):
    file_name: str
    file_path: Optional[str] = None
    object_name: Optional[str] = None
    bucket_name: Optional[str] = None
    namespace_name: Optional[str] = None
    module_code: Optional[str] = None
    doc_type_code: Optional[str] = None
    mime_type: Optional[str] = None
    content_text: Optional[str] = None


class IngestionApiRequest(BaseModel):
    requested_by: str = Field(..., description="User or system triggering ingestion")
    source_system: str = Field(default="APEX", description="Source system name")
    sync_bucket: bool = False
    anonymize_docs: bool = True
    chunk_max_tokens: int = 350
    chunk_overlap_tokens: int = 30
    project: ProjectPayload
    documents: List[DocumentPayload]


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_docs_folder() -> str:
    return str(_backend_root() / "app" / "data" / "downloads")


def _extract_document_text(doc: DocumentPayload) -> str:
    if doc.content_text and doc.content_text.strip():
        return doc.content_text.strip()

    source_path = doc.file_path
    if not source_path and doc.object_name:
        source_path = str(Path(get_docs_folder()) / doc.object_name)

    if not source_path:
        raise ValueError(f"No content_text or file_path/object_name provided for {doc.file_name}")

    return extract_text_with_formatting_in_sequence(source_path)


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
    doc: DocumentPayload,
    *,
    batch_id: str,
    run_id: str,
    project_id: str,
    requested_by: str,
    anonymize_docs: bool,
    chunk_max_tokens: int,
    chunk_overlap_tokens: int,
) -> Dict[str, Any]:
    object_name = doc.object_name or doc.file_name
    bucket_name = doc.bucket_name or DEFAULT_BUCKET
    namespace_name = doc.namespace_name or DEFAULT_NAMESPACE
    object_uri = compute_object_uri(namespace_name, bucket_name, object_name)
    mime_type = doc.mime_type or mimetypes.guess_type(doc.file_name)[0]

    document_id = create_document_record(
        project_id=project_id,
        batch_id=batch_id,
        file_name=doc.file_name,
        requested_by=requested_by,
        object_name=object_name,
        bucket_name=bucket_name,
        namespace_name=namespace_name,
        object_uri=object_uri,
        module_code=doc.module_code,
        doc_type_code=doc.doc_type_code,
        mime_type=mime_type,
    )

    try:
        extracted_text = _extract_document_text(doc)
        processed_text = anonymize_markdown(extracted_text) if anonymize_docs else extracted_text

        chunk_rows = _chunk_text(processed_text, chunk_max_tokens, chunk_overlap_tokens)
        prepared_chunks: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunk_rows):
            chunk_text = chunk.get("chunk", "")
            prepared_chunks.append(
                {
                    "chunk_id": make_deterministic_chunk_id(document_id, idx, chunk_text),
                    "chunk_index": idx,
                    "heading": chunk.get("heading"),
                    "source_file": doc.file_name,
                    "chunk": chunk_text,
                }
            )

        embedded_chunks = _embed_chunks(prepared_chunks)
        stored_chunks = insert_document_embeddings(
            document_id=document_id,
            run_id=run_id,
            chunks=embedded_chunks,
            created_by=requested_by,
        )

        mark_document_status(document_id, "COMPLETED", requested_by)
        return {
            "document_id": document_id,
            "file_name": doc.file_name,
            "status": "COMPLETED",
            "chunks_created": len(prepared_chunks),
            "vectors_stored": stored_chunks,
        }
    except Exception as exc:
        mark_document_status(document_id, "FAILED", requested_by)
        return {
            "document_id": document_id,
            "file_name": doc.file_name,
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


@router.post("/ingest")
def run_ingestion(payload: IngestionApiRequest):
    if not payload.documents:
        raise HTTPException(status_code=400, detail="At least one document is required.")

    pipeline_started_at = time.perf_counter()

    try:
        project = get_or_create_project(payload.project.model_dump(), payload.requested_by)

        if payload.sync_bucket:
            download_all_from_bucket()

        batch_id = create_ingestion_batch(payload.source_system, payload.requested_by, len(payload.documents))
        run_id = create_ingestion_run(batch_id, payload.requested_by, len(payload.documents))

        results = []
        successful_documents = 0
        failed_documents = 0

        for doc in payload.documents:
            result = _process_document(
                doc,
                batch_id=batch_id,
                run_id=run_id,
                project_id=project["project_id"],
                requested_by=payload.requested_by,
                anonymize_docs=payload.anonymize_docs,
                chunk_max_tokens=payload.chunk_max_tokens,
                chunk_overlap_tokens=payload.chunk_overlap_tokens,
            )
            results.append(result)
            if result["status"] == "COMPLETED":
                successful_documents += 1
            else:
                failed_documents += 1

        finalize_ingestion_run(run_id, successful_documents, failed_documents, payload.requested_by)
        finalize_ingestion_batch(batch_id, successful_documents, failed_documents, payload.requested_by)

        return {
            "message": "Ingestion pipeline completed.",
            "project_id": project["project_id"],
            "project_name": project["project_name"],
            "batch_id": batch_id,
            "run_id": run_id,
            "successful_documents": successful_documents,
            "failed_documents": failed_documents,
            "time_taken_seconds": round(time.perf_counter() - pipeline_started_at, 3),
            "documents": results,
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
