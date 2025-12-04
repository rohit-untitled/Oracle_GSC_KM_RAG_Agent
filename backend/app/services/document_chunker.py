import spacy
import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import unicodedata
import re
import logging

# Load small English NLP model once
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_sensitive_terms(text: str) -> list:
    """Extract potential sensitive terms (ORG, PERSON, GPE, etc.) from text."""
    terms = set()

    # NLP entity extraction
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT"]:
            terms.add(ent.text)

    # Regex patterns (emails, phone numbers, codes)
    emails = re.findall(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', text, flags=re.I)
    phones = re.findall(r'\b\d{10}\b', text)
    codes = re.findall(r'\b[A-Z]{2,}\d+\b', text)

    terms.update(emails + phones + codes)
    return list(terms)

def clean_text(text: str) -> str:
    """
    Cleans and normalizes text to ensure consistency across documents.
    Handles encoding artifacts, irregular whitespace, and non-printable characters.
    """
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    # Replace multiple newlines with two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Strip non-printable characters
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]+", "", text)
    return text.strip()


def chunk_document(
    text: str,
    file_path: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Splits a single cleaned document into overlapping, semantically consistent chunks.

    Args:
        text: Document text.
        file_path: Source file path (for traceability).
        chunk_size: Max characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of dicts containing chunk text and metadata.
    """
    text = clean_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
        length_function=len,
        is_separator_regex=False
    )

    chunks = splitter.split_text(text)

    chunked_docs = [
        {
            "file_path": file_path,
            "chunk_index": idx,
            "text": chunk,
            "sensitive_terms": extract_sensitive_terms(chunk),  # NEW
            "metadata": {
                "source": file_path.split("\\")[-1],
                "chunk_length": len(chunk),
                "total_chunks": len(chunks)
            },
        }
        for idx, chunk in enumerate(chunks)
    ]


    logger.info(f"✅ {file_path} → {len(chunks)} chunks created.")
    return chunked_docs


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 1200,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Processes multiple documents and returns chunked results with metadata.

    Args:
        documents: List of {"file_path": ..., "text": ...}
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunked documents.
    """
    all_chunks = []
    for doc in documents:
        try:
            chunks = chunk_document(
                text=doc["text"],
                file_path=doc["file_path"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"❌ Error chunking {doc['file_path']}: {str(e)}")

    logger.info(f"📄 Total chunks created: {len(all_chunks)}")
    return all_chunks
