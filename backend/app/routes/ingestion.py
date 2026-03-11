import hashlib
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.services.anonymize_service import anonymize_markdown_files
from app.services.chunk_service import chunk_anonymized_documents
from app.services.docx_extractor import extract_text_with_formatting_in_sequence
from app.services.document_loader import load_docx_files
from app.services.embedding_service import OCIEmbeddingService
from app.services.oci_downloader import download_all_from_bucket
from app.services.vector_store_service import insert_embeddings_from_json


router = APIRouter()
logger = logging.getLogger("KM Knowledge Agent is Working")


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_docs_folder() -> str:
    return str(_backend_root() / "app" / "data" / "downloads")


def get_extracted_folder() -> str:
    return str(_backend_root() / "app" / "data" / "extracted")


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


@router.get("/extract-docs")
def extract_docs():
    docs = load_docx_files(get_docs_folder())
    output_dir = Path(get_extracted_folder())
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for doc in docs:
        try:
            file_base = Path(doc["file_path"]).name
            stem = Path(file_base).stem
            ext = Path(file_base).suffix
            out_name = f"{stem}{ext.lower().replace('.', '_')}.md"
            out_path = output_dir / out_name
            if out_path.exists():
                result[file_base] = f"Skipped (already extracted): {out_path}"
                continue

            text = extract_text_with_formatting_in_sequence(doc["file_path"])
        except Exception as e:
            text = f"Error extracting: {e}"

        result[file_base] = text
        try:
            out_path.write_text(text, encoding="utf-8")
        except Exception as e:
            result[file_base] = f"Error writing file: {e}"

    return result


@router.get("/anonymize-docs")
def anonymize_docs():
    return anonymize_markdown_files(get_extracted_folder())


@router.get("/chunk-anonymized")
def chunk_anonymized():
    base_dir = _backend_root() / "app" / "data"
    base_dir.mkdir(parents=True, exist_ok=True)
    return chunk_anonymized_documents(str(base_dir))


@router.post("/embed-chunks")
def embed_chunks():
    chunks_path = _backend_root() / "app" / "data" / "chunks" / "chunks.json"
    output_path = _backend_root() / "app" / "data" / "chunks" / "chunks_with_embeddings.json"

    if not chunks_path.exists():
        return {"error": "chunks.json not found. Run /chunk-anonymized first."}

    embedder = OCIEmbeddingService()
    start_time = time.time()

    try:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load {chunks_path}: {e}")
        return {"error": "Failed to load chunks.json"}

    output = []
    existing = {}

    if output_path.exists():
        try:
            previous = json.loads(output_path.read_text(encoding="utf-8"))
            for item in previous:
                cid = item.get("chunk_id")
                if cid and item.get("embedding"):
                    existing[cid] = item
        except Exception as e:
            logger.warning(f"Failed to load existing embeddings for resume: {e}")

    total_chunks = len(chunks)
    successful = 0
    empty_vectors = 0
    split_depth_counts = {}

    batch_texts = []
    batch_meta = []

    def _flush_batch():
        nonlocal batch_texts, batch_meta, output, successful, empty_vectors
        if not batch_texts:
            return
        vectors = embedder.embed_texts(batch_texts)
        for (idx, ch, _), emb in zip(batch_meta, vectors):
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
            logger.warning(f"Chunk {idx} is empty - skipping embedding")
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
                for (i2, ch2, _) in batch_meta:
                    try:
                        emb, depth = embedder.embed_text(ch2.get("chunk", ""), return_depth=True)
                        split_depth_counts[depth] = split_depth_counts.get(depth, 0) + 1
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
                    split_depth_counts[depth] = split_depth_counts.get(depth, 0) + 1
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

    try:
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to write embeddings JSON: {e}")
        return {"error": "Failed to write chunks_with_embeddings.json"}

    total_time = round(time.time() - start_time, 2)
    return {
        "message": "Embeddings created successfully",
        "file": output_path.name,
        "stats": {
            "total_chunks": total_chunks,
            "successful_embeddings": successful,
            "empty_vectors": empty_vectors,
            "split_depth_counts": split_depth_counts,
            "time_taken_seconds": total_time,
        },
    }


@router.post("/store-embeddings")
def store_embeddings_endpoint():
    json_file = _backend_root() / "app" / "data" / "chunks" / "chunks_with_embeddings.json"

    if not json_file.exists():
        return {"error": "chunks_with_embeddings.json not found. Run /embed-chunks first."}

    try:
        inserted = insert_embeddings_from_json(str(json_file))
    except Exception as e:
        logger.error(f"Failed to store embeddings in vector DB: {e}")
        return {"error": "Failed to store embeddings"}

    return {"status": "ok", "inserted_records": inserted}
