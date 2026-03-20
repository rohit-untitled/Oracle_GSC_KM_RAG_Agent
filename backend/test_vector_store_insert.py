import json
import os
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env_file(Path(__file__).resolve().parents[1] / ".env")

from app.services.vector_store_service import insert_embedding_payload, search_similar_chunks  # noqa: E402


def main() -> None:
    sample_payload = {
        "document_id": "TEST_DOC_001",
        "run_id": "TEST_RUN_001",
        "chunk_id": "TEST_CHUNK_001",
        "chunk_index": 1,
        "heading": "Vector Store Test Heading",
        "source_file": "sample_test_document.docx",
        "chunk": "This is a sample chunk inserted to verify the new Oracle KM vector store schema.",
        "embedding": [0.001] * 1536,
        "created_by": "KM_RAG_AGENT_TEST",
    }

    print("Inserting sample payload into new KM vector schema...")
    result = insert_embedding_payload(sample_payload)
    print("Insert result:")
    print(json.dumps(result, indent=2))

    print("\nRunning similarity search using the same sample vector...")
    hits = search_similar_chunks(sample_payload["embedding"], top_k=3)
    print(json.dumps(hits, indent=2, default=str))


if __name__ == "__main__":
    main()