import os
import json
import nltk

nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

def token_len(text: str) -> int:
    return len(word_tokenize(text))

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

def chunk_anonymized_documents(base_dir: str, max_tokens: int = 450):
    """
    Creates chunks from anonymized text files:

    ✔ Balanced chunk size (< 450 tokens)
    ✔ Splits long sentences recursively
    ✔ No chunk ever exceeds OCI embedding limit
    ✔ Chunks preserve sentence boundaries where possible
    """

    anonymized_dir = os.path.join(base_dir, "anonymized")
    chunk_dir = os.path.join(base_dir, "chunks")

    os.makedirs(chunk_dir, exist_ok=True)

    all_chunks = []

    # Loop through all anonymized .txt files
    for filename in os.listdir(anonymized_dir):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(anonymized_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        # Sentence tokenize
        sentences = sent_tokenize(text)

        current_chunk_sentences = []
        current_tokens = 0

        # Process each sentence
        for sent in sentences:
            sent = sent.strip()
            sent_tokens = token_len(sent)

            # ---- Case 1: Sentence itself is too long → split recursively
            if sent_tokens > max_tokens:
                small_parts = split_sentence_recursive(sent, max_tokens)

                # Add each tiny part individually
                for part in small_parts:
                    part_tokens = token_len(part)

                    # If current chunk can't hold this part → flush chunk
                    if current_tokens + part_tokens > max_tokens:
                        all_chunks.append({
                            "source_file": filename,
                            "chunk": " ".join(current_chunk_sentences)
                        })
                        current_chunk_sentences = []
                        current_tokens = 0

                    current_chunk_sentences.append(part)
                    current_tokens += part_tokens

                continue  # move to next sentence

            # ---- Case 2: Normal-size sentence
            if current_tokens + sent_tokens > max_tokens:
                # Flush the full chunk
                all_chunks.append({
                    "source_file": filename,
                    "chunk": " ".join(current_chunk_sentences)
                })
                current_chunk_sentences = []
                current_tokens = 0

            # Add sentence
            current_chunk_sentences.append(sent)
            current_tokens += sent_tokens

        # Add any remaining chunk
        if current_chunk_sentences:
            all_chunks.append({
                "source_file": filename,
                "chunk": " ".join(current_chunk_sentences)
            })

    # Save chunks.json
    output_file = os.path.join(chunk_dir, "chunks.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    return {
        "message": "Token-safe chunking completed",
        "total_chunks": len(all_chunks),
        "output_file": output_file
    }
