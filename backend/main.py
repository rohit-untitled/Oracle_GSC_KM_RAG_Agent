from pathlib import Path
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)

import os
import logging
import asyncio
import json
import time
import hashlib
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional

# ---- Internal imports ----
from app.services.document_loader import load_docx_files
from app.services.docx_extractor import extract_text_with_formatting_in_sequence
from app.services.document_chunker import chunk_documents
from app.services.chunk_service import chunk_anonymized_documents
from app.services.anonymize_service import anonymize_markdown_files
from app.services.embedding_service import OCIEmbeddingService
from app.services.rag_service import answer_query
from app.services.session_store_service import (
    fetch_session_history,
    delete_session_history,
    ensure_session,
    get_session,
    list_sessions,
    delete_session,
)
from app.services.vector_store_service import insert_embeddings_from_json
from app.services.oci_downloader import download_all_from_bucket


# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("KM Knowledge Agent is Working")

app = FastAPI(
    title="AI Redaction Agent",
    description="RAG-powered system for Q&A and redaction with multi-turn memory",
    version="2.0.0",
)

def get_docs_folder() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "app",
        "data",
        "downloads"
    )

def get_extracted_folder() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "app",
        "data",
        "extracted"
    )

# Health check
@app.get("/")
def root():
    return {"message": "AI Redaction Agent is running!"}

# Sync docs from OCI bucket
@app.get("/sync-bucket")
def sync_bucket():
    try:
        download_all_from_bucket()
        return {"message": "All documents downloaded from OCI."}
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(500, str(e))

# List supported docs from downloads (.docx/.pptx/.txt)
@app.get("/load-docs")
def load_docs():
    folder = get_docs_folder()
    docs = load_docx_files(folder)

    return {
        "total_documents": len(docs),
        "documents": [
            {
                "file": doc["file_name"],
                "folder": doc["folder"],
                "path": doc["file_path"]
            }
            for doc in docs
        ]
    }


# Extract supported docs to markdown
@app.get("/extract-docs")
def extract_docs():
    folder = get_docs_folder()
    docs = load_docx_files(folder)
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "app",
        "data",
        "extracted"
    )
    os.makedirs(output_dir, exist_ok=True)

    result = {}
    for doc in docs:
        try:
            file_base = os.path.basename(doc["file_path"])
            stem, ext = os.path.splitext(file_base)
            out_name = f"{stem}{ext.lower().replace('.', '_')}.md"
            out_path = os.path.join(output_dir, out_name)
            if os.path.exists(out_path):
                result[os.path.basename(doc["file_path"])] = f"Skipped (already extracted): {out_path}"
                continue

            text = extract_text_with_formatting_in_sequence(doc["file_path"])
        except Exception as e:
            text = f"Error extracting: {e}"

        result[os.path.basename(doc["file_path"])] = text
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            result[os.path.basename(doc["file_path"])] = f"Error writing file: {e}"

    return result


# Anonymize extracted markdown files
@app.get("/anonymize-docs")
def anonymize_docs():
    extracted_dir = get_extracted_folder()
    return anonymize_markdown_files(extracted_dir)

# Chunk anonymized markdown content
@app.get("/chunk-anonymized")
def chunk_anonymized():
    base_dir = os.path.join(os.path.dirname(__file__),"app", "data")

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    result = chunk_anonymized_documents(base_dir)
    return result


# Embed chunks and persist embeddings JSON
@app.post("/embed-chunks")
def embed_chunks():
    chunks_path = os.path.join(
        os.path.dirname(__file__),"app", "data", "chunks", "chunks.json"
    )
    output_path = os.path.join(
        os.path.dirname(__file__),"app", "data", "chunks", "chunks_with_embeddings.json"
    )

    if not os.path.exists(chunks_path):
        return {
            "error": "chunks.json not found. Run /chunk-anonymized first."
        }

    embedder = OCIEmbeddingService()
    start_time = time.time()

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {chunks_path}: {e}")
        return {"error": "Failed to load chunks.json"}

    output = []
    existing = {}

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                previous = json.load(f)
            for item in previous:
                cid = item.get("chunk_id")
                if cid and item.get("embedding"):
                    existing[cid] = item
        except Exception as e:
            logger.warning(f"Failed to load existing embeddings for resume: {e}")

    # ---- Summary tracking ----
    total_chunks = len(chunks)
    successful = 0
    empty_vectors = 0
    split_depth_counts = {}   # {depth: count}

    batch_texts = []
    batch_meta = []

    def _flush_batch():
        nonlocal batch_texts, batch_meta, output, successful, empty_vectors, split_depth_counts
        if not batch_texts:
            return
        vectors = embedder.embed_texts(batch_texts)
        for (idx, ch, cid), emb in zip(batch_meta, vectors):
            if not emb:
                logger.error(f"[Chunk {idx}] Empty vector returned")
                empty_vectors += 1
                ch["embedding"] = []
            else:
                successful += 1
                ch["embedding"] = emb
            output.append(ch)
        batch_texts = []
        batch_meta = []

    for idx, ch in enumerate(chunks, start=1):
        text = ch.get("chunk", "").strip()

        if not text:
            logger.warning(f"Chunk {idx} is empty — skipping embedding")
            ch["embedding"] = []
            empty_vectors += 1
            output.append(ch)
            continue

        source_key = ch.get("source_file", "") + "|" + str(ch.get("chunk_index", idx))
        cid = hashlib.sha256((source_key + "|" + text).encode("utf-8")).hexdigest()
        ch["chunk_id"] = cid

        if cid in existing:
            output.append(existing[cid])
            continue

        batch_texts.append(text)
        batch_meta.append((idx, ch, cid))

        if len(batch_texts) >= 16:
            try:
                _flush_batch()
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Fallback to per-item
                for (i2, ch2, _) in batch_meta:
                    try:
                        emb, depth = embedder.embed_text(ch2.get("chunk", ""), return_depth=True)
                        if depth not in split_depth_counts:
                            split_depth_counts[depth] = 0
                        split_depth_counts[depth] += 1
                        if not emb:
                            empty_vectors += 1
                            ch2["embedding"] = []
                        else:
                            successful += 1
                            ch2["embedding"] = emb
                    except Exception as e2:
                        logger.error(f"Exception while embedding chunk {i2}: {e2}")
                        ch2["embedding"] = []
                        empty_vectors += 1
                    output.append(ch2)
                batch_texts = []
                batch_meta = []

    if batch_texts:
        try:
            _flush_batch()
        except Exception as e:
            logger.error(f"Final batch embedding failed: {e}")
            for (i2, ch2, _) in batch_meta:
                try:
                    emb, depth = embedder.embed_text(ch2.get("chunk", ""), return_depth=True)
                    if depth not in split_depth_counts:
                        split_depth_counts[depth] = 0
                    split_depth_counts[depth] += 1
                    if not emb:
                        empty_vectors += 1
                        ch2["embedding"] = []
                    else:
                        successful += 1
                        ch2["embedding"] = emb
                except Exception as e2:
                    logger.error(f"Exception while embedding chunk {i2}: {e2}")
                    ch2["embedding"] = []
                    empty_vectors += 1
                output.append(ch2)

    # ---- Save final embeddings ----
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write embeddings JSON: {e}")
        return {"error": "Failed to write chunks_with_embeddings.json"}

    total_time = round(time.time() - start_time, 2)

    # ---- FINAL SUMMARY ----
    summary = {
        "message": "Embeddings created successfully",
        "file": os.path.basename(output_path),
        "stats": {
            "total_chunks": total_chunks,
            "successful_embeddings": successful,
            "empty_vectors": empty_vectors,
            "split_depth_counts": split_depth_counts,
            "time_taken_seconds": total_time,
        }
    }

    return summary


# Store embeddings into vector DB
@app.post("/store-embeddings")
def store_embeddings_endpoint():
    json_file = os.path.join(
        os.path.dirname(__file__), "app", "data", "chunks", "chunks_with_embeddings.json"
    )

    if not os.path.exists(json_file):
        return {
            "error": "chunks_with_embeddings.json not found. Run /embed-chunks first."
        }

    try:
        inserted = insert_embeddings_from_json(json_file)
    except Exception as e:
        logger.error(f"Failed to store embeddings in vector DB: {e}")
        return {"error": "Failed to store embeddings"}

    return {
        "status": "ok",
        "inserted_records": inserted
    }


class RAGRequest(BaseModel):
    query: str
    top_k: Optional[int] = Field(5, ge=1, le=20)
    session_id: Optional[str] = None


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None

# Fetch chat history by session id
@app.get("/session-history")
def session_history_api(session_id: str):
    return fetch_session_history(session_id)

# List chat sessions for sidebar
@app.get("/sessions")
def list_sessions_api(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    return list_sessions(limit=limit, offset=offset)

# Get one chat session metadata
@app.get("/sessions/{session_id}")
def get_session_api(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

# Delete chat session and its history
@app.delete("/sessions/{session_id}")
def delete_session_api(session_id: str):
    result = delete_session(session_id)
    if result["session_deleted"] == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "status": "ok",
        "session_id": session_id,
        "history_deleted": result["history_deleted"],
        "session_deleted": result["session_deleted"],
    }

# Delete chat history by session id
@app.delete("/session-history/{session_id}")
def delete_session_history_api(session_id: str):
    deleted = delete_session_history(session_id)
    return {
        "status": "ok",
        "session_id": session_id,
        "deleted_records": deleted
    }

# Create a new chat session (optional title)
@app.post("/sessions")
def create_session(payload: Optional[CreateSessionRequest] = None):
    session_id = str(uuid.uuid4())
    title = payload.title.strip() if payload and payload.title else None
    ensure_session(session_id, title=title)
    return {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

# Ask a question with auto-session creation
@app.post("/ask")
def ask_endpoint(payload: RAGRequest):
    """
    Multi-turn RAG Q&A endpoint.
    """
    session_id = payload.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        ensure_session(session_id)
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    logger.info(f"Session {session_id}: Received query")

    try:
        response = answer_query(
            query=query,
            top_k=payload.top_k or 5,
            session_id=session_id
        )

        return {
            "session_id": session_id,
            "answer": response["answer"],
            "chunks": response["chunks"],
            "history_length": response["history_length"]
        }

    except Exception as e:
        logger.exception(f"Error in RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
