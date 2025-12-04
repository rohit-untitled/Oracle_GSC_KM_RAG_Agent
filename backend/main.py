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
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, conint
from typing import List, Optional

# ---- Internal imports ----
from app.services.document_loader import load_docx_files
from app.services.docx_extractor import extract_text_with_formatting_in_sequence
from app.services.document_chunker import chunk_documents
from app.services.chunk_service import chunk_anonymized_documents
from app.services.anonymize_service import anonymize_documents
from app.services.embedding_service import OCIEmbeddingService
from app.services.rag_service import answer_query, ai_redact_sensitive_info
from app.services.vector_store_service import insert_embeddings_from_json
from app.services.oci_downloader import download_all_from_bucket
from app.services.vector_store_service import init_oracle_client
init_oracle_client()


# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai_redaction_agent")

app = FastAPI(
    title="AI Redaction Agent",
    description="RAG-powered system for privacy-safe document Q&A and redaction",
    version="1.0.0",
)


def get_docs_folder() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "app",
        "data",
        "downloads"
    )

@app.get("/")
def root():
    return {"message": "AI Redaction Agent is running!"}

@app.get("/sync-bucket")
def sync_bucket():
    try:
        download_all_from_bucket()
        return {"message": "All documents downloaded from OCI."}
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(500, str(e))

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



@app.get("/extract-docs")
def extract_docs():
    folder = get_docs_folder()
    docs = load_docx_files(folder)

    result = {}
    for doc in docs:
        try:
            text = extract_text_with_formatting_in_sequence(doc["file_path"])
        except Exception as e:
            text = f"Error extracting: {e}"

        result[os.path.basename(doc["file_path"])] = text

    return result


@app.get("/anonymize-docs")
def anonymize_docs():
    folder = get_docs_folder()
    return anonymize_documents(folder)

@app.get("/chunk-anonymized")
def chunk_anonymized():
    base_dir = os.path.join(os.path.dirname(__file__),"app", "data")

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    result = chunk_anonymized_documents(base_dir)
    return result


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

    # ---- Summary tracking ----
    total_chunks = len(chunks)
    successful = 0
    empty_vectors = 0
    split_depth_counts = {}   # {depth: count}

    for idx, ch in enumerate(chunks, start=1):
        text = ch.get("chunk", "").strip()

        if not text:
            logger.warning(f"Chunk {idx} is empty — skipping embedding")
            ch["embedding"] = []
            empty_vectors += 1
            output.append(ch)
            continue

        try:
            # ---- Get embedding + depth info ----
            emb, depth = embedder.embed_text(text, return_depth=True)

            # Track depth usage
            if depth not in split_depth_counts:
                split_depth_counts[depth] = 0
            split_depth_counts[depth] += 1

            if not emb:
                logger.error(f"[Chunk {idx}] Empty vector returned")
                empty_vectors += 1
            else:
                successful += 1

            ch["embedding"] = emb

        except Exception as e:
            logger.error(f"Exception while embedding chunk {idx}: {e}")
            ch["embedding"] = []
            empty_vectors += 1

        output.append(ch)

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
    top_k: Optional[int] = 5

@app.post("/ask")
def ask_endpoint(payload: RAGRequest):
    """
    Query the RAG system and get LLM answer
    """
    try:
        response = answer_query(payload.query, top_k=payload.top_k)
        return response
    except Exception as e:
        logger.exception(f"Error in RAG query: {e}")
        return {"error": str(e)}