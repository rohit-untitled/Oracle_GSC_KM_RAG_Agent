import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    tiktoken = None

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize

_TOKENIZER_ENCODING = None
if tiktoken is not None:
    try:
        _TOKENIZER_ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKENIZER_ENCODING = None

def token_len(text: str) -> int:
    value = (text or "").strip()
    if not value:
        return 0
    if _TOKENIZER_ENCODING is not None:
        return len(_TOKENIZER_ENCODING.encode(value))
    return max(1, len(value) // 4)

def split_sentence_recursive(sentence: str, max_tokens: int):
    tokens = word_tokenize(sentence)

    if len(tokens) <= max_tokens:
        return [sentence]

    # Split sentence into two halves
    mid = len(tokens) // 2
    part1 = " ".join(tokens[:mid])
    part2 = " ".join(tokens[mid:])

    # Recurse until all parts are small enough
    return (
        split_sentence_recursive(part1, max_tokens) +
        split_sentence_recursive(part2, max_tokens)
    )

def _is_table_line(line: str) -> bool:
    return line.count("|") >= 2


def _is_table_alignment_row(line: str) -> bool:
    stripped = line.strip().strip("|").strip()
    if not stripped:
        return False
    return bool(re.fullmatch(r"[:\-\s]+", stripped))


def _parse_markdown_blocks(md_text: str) -> List[Dict[str, Any]]:
    """
    Convert markdown into ordered blocks while preserving structure.
    """
    lines = md_text.splitlines()
    blocks: List[Dict[str, Any]] = []
    current: List[str] = []
    current_heading: Optional[str] = None
    current_heading_level: Optional[int] = None
    in_code_fence = False
    block_index = 0

    def flush(kind: str = "paragraph"):
        nonlocal current, block_index
        if current:
            blocks.append({
                "block_index": block_index,
                "type": kind,
                "heading": current_heading or "",
                "heading_level": current_heading_level,
                "text": "\n".join(current).strip()
            })
            block_index += 1
            current = []

    for line in lines:
        if line.strip().startswith("```"):
            if in_code_fence:
                current.append(line)
                flush("code")
                in_code_fence = False
            else:
                flush()
                in_code_fence = True
                current.append(line)
            continue

        if in_code_fence:
            current.append(line)
            continue

        if re.match(r"^#{1,6}\s+", line):
            flush()
            current_heading = line.strip()
            current_heading_level = len(line) - len(line.lstrip("#"))
            blocks.append({
                "block_index": block_index,
                "type": "heading",
                "heading": current_heading,
                "heading_level": current_heading_level,
                "text": current_heading
            })
            block_index += 1
            continue

        if _is_table_line(line):
            if current:
                flush()
            table_lines = [line]
            # consume consecutive table lines handled in caller loop by appending
            blocks.append({
                "block_index": block_index,
                "type": "table",
                "heading": current_heading or "",
                "heading_level": current_heading_level,
                "text": "\n".join(table_lines).strip()
            })
            block_index += 1
            continue

        if not line.strip():
            flush()
            continue

        current.append(line)

    flush()
    return blocks


def _merge_consecutive_tables(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for blk in blocks:
        if merged and blk["type"] == "table" and merged[-1]["type"] == "table":
            merged[-1]["text"] = (merged[-1]["text"] + "\n" + blk["text"]).strip()
            merged[-1]["block_index_end"] = blk.get("block_index", merged[-1].get("block_index"))
        else:
            blk = dict(blk)
            blk.setdefault("block_index_end", blk.get("block_index"))
            merged.append(blk)
    return merged


def _split_block_text(text: str, max_tokens: int) -> List[str]:
    sentences = sent_tokenize(text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tokens = token_len(sent)

        if sent_tokens > max_tokens:
            parts = split_sentence_recursive(sent, max_tokens)
            for part in parts:
                part_tokens = token_len(part)
                if current_tokens + part_tokens > max_tokens and current:
                    chunks.append(" ".join(current).strip())
                    current = []
                    current_tokens = 0
                current.append(part)
                current_tokens += part_tokens
            continue

        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current).strip())
            current = []
            current_tokens = 0

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def _chunk_blocks(
    blocks: List[Dict[str, Any]],
    max_tokens: int,
    overlap_tokens: int
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    buffer_texts: List[str] = []
    buffer_tokens = 0
    current_heading = ""
    current_heading_level: Optional[int] = None
    buffer_types: List[str] = []
    block_index_start: Optional[int] = None
    block_index_end: Optional[int] = None

    def _resolve_chunk_type(types: List[str]) -> str:
        normalized = [t for t in types if t and t != "heading"]
        if not normalized:
            return "heading"
        unique = set(normalized)
        if len(unique) == 1:
            return normalized[0]
        return "mixed"

    def flush():
        nonlocal buffer_texts, buffer_tokens, buffer_types, block_index_start, block_index_end
        if not buffer_texts:
            return
        chunk_text = "\n".join(buffer_texts).strip()
        chunk_type = _resolve_chunk_type(buffer_types)
        token_count = token_len(chunk_text) if chunk_text else 0
        chunks.append({
            "heading": current_heading,
            "chunk": chunk_text,
            "chunk_type": chunk_type,
            "heading_level": current_heading_level,
            "token_count": token_count,
            "contains_table": "table" in buffer_types,
            "contains_code": "code" in buffer_types,
            "block_index_start": block_index_start,
            "block_index_end": block_index_end,
            "metadata": {
                "chunk_type": chunk_type,
                "heading": current_heading,
                "heading_level": current_heading_level,
                "token_count": token_count,
                "overlap_tokens": overlap_tokens,
                "contains_table": "table" in buffer_types,
                "contains_code": "code" in buffer_types,
                "block_index_start": block_index_start,
                "block_index_end": block_index_end,
                "parser": "markdown_structure_v2",
            },
        })
        if overlap_tokens > 0:
            # simple overlap by keeping last ~overlap_tokens words
            words = word_tokenize(chunk_text)
            tail = " ".join(words[-overlap_tokens:]) if words else ""
            buffer_texts = [tail] if tail else []
            buffer_tokens = token_len(tail) if tail else 0
            buffer_types = [chunk_type] if tail else []
            block_index_start = block_index_end
        else:
            buffer_texts = []
            buffer_tokens = 0
            buffer_types = []
            block_index_start = None
            block_index_end = None

    for blk in blocks:
        if blk["type"] == "heading":
            flush()
            current_heading = blk["text"]
            current_heading_level = blk.get("heading_level")
            buffer_texts = [blk["text"]]
            buffer_tokens = token_len(blk["text"])
            buffer_types = ["heading"]
            block_index_start = blk.get("block_index")
            block_index_end = blk.get("block_index")
            continue

        blk_text = blk["text"]
        if blk["type"] in ("table", "code"):
            blk_tokens = token_len(blk_text)
            if buffer_tokens + blk_tokens > max_tokens and buffer_texts:
                flush()
            if block_index_start is None:
                block_index_start = blk.get("block_index")
            buffer_texts.append(blk_text)
            buffer_tokens += blk_tokens
            buffer_types.append(blk["type"])
            block_index_end = blk.get("block_index_end", blk.get("block_index"))
            continue

        # paragraph/list blocks
        parts = _split_block_text(blk_text, max_tokens)
        for part in parts:
            part_tokens = token_len(part)
            if buffer_tokens + part_tokens > max_tokens and buffer_texts:
                flush()
            if block_index_start is None:
                block_index_start = blk.get("block_index")
            buffer_texts.append(part)
            buffer_tokens += part_tokens
            buffer_types.append(blk["type"])
            block_index_end = blk.get("block_index_end", blk.get("block_index"))

    flush()
    return chunks


def chunk_anonymized_documents(base_dir: str, max_tokens: int = 350, overlap_tokens: int = 30):

    anonymized_dir = os.path.join(base_dir, "anonymized")
    chunk_dir = os.path.join(base_dir, "chunks")

    os.makedirs(chunk_dir, exist_ok=True)

    output_file = os.path.join(chunk_dir, "chunks.json")

    existing_chunks: List[Dict[str, str]] = []
    processed_sources = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
            for ch in existing_chunks:
                if ch.get("source_md"):
                    processed_sources.add(ch["source_md"])
        except Exception:
            existing_chunks = []
            processed_sources = set()

    all_chunks: List[Dict[str, str]] = list(existing_chunks)

    # Loop through all anonymized .md files
    for filename in os.listdir(anonymized_dir):
        if not filename.lower().endswith(".md"):
            continue
        if filename in processed_sources:
            continue

        file_path = os.path.join(anonymized_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        blocks = _parse_markdown_blocks(text)
        blocks = _merge_consecutive_tables(blocks)
        chunks = _chunk_blocks(blocks, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

        source_name = Path(filename).stem
        if source_name.endswith(("_docx", "_pptx", "_pdf", "_txt")):
            source_file = f"{source_name.rsplit('_', 1)[0]}.{source_name.rsplit('_', 1)[1]}"
        else:
            source_file = source_name
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "source_file": source_file,
                "source_md": filename,
                "chunk_index": i,
                "heading": ch["heading"],
                "chunk": ch["chunk"]
            })

    # Save chunks.json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    return {
        "message": "Token-safe chunking completed",
        "total_chunks": len(all_chunks),
        "output_file": output_file
    }
