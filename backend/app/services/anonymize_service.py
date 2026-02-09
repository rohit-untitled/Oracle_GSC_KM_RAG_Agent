import os
import re
from typing import Dict, List, Optional, Tuple
from app.services.rag_service import ai_redact_sensitive_info_batch


def _is_table_alignment_row(line: str) -> bool:
    stripped = line.strip().strip("|").strip()
    if not stripped:
        return False
    return bool(re.fullmatch(r"[:\-\s]+", stripped))


def _batch_anonymize_texts(
    texts: List[str],
    cache: Optional[Dict[str, str]] = None,
    max_chars: int = 6000
) -> List[str]:
    if cache is None:
        cache = {}

    results: List[str] = ["" for _ in texts]
    pending: List[Tuple[int, str]] = []

    for i, t in enumerate(texts):
        t_norm = t.strip()
        if not t_norm:
            results[i] = t_norm
            continue
        if t_norm in cache:
            results[i] = cache[t_norm]
            continue
        pending.append((i, t_norm))

    if not pending:
        return results

    batch: List[str] = []
    batch_indices: List[int] = []
    batch_len = 0

    def _flush_batch():
        nonlocal batch, batch_indices, batch_len
        if not batch:
            return
        anonymized = ai_redact_sensitive_info_batch(batch)
        for idx, anon in zip(batch_indices, anonymized):
            cache[ texts[idx].strip() ] = anon
            results[idx] = anon
        batch = []
        batch_indices = []
        batch_len = 0

    for idx, t in pending:
        if batch_len + len(t) > max_chars and batch:
            _flush_batch()
        batch.append(t)
        batch_indices.append(idx)
        batch_len += len(t)

    _flush_batch()
    return results


def anonymize_markdown(md_text: str, cache: Optional[Dict[str, str]] = None) -> str:
    """
    Anonymize markdown text while preserving structure and sequence.
    """
    lines = md_text.splitlines()
    out_lines = []
    in_code_fence = False
    segments: List[str] = []
    builders: List[Tuple[int, str, dict]] = []

    for line in lines:
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            out_lines.append(line)
            continue

        if in_code_fence:
            out_lines.append(line)
            continue

        if not line.strip():
            out_lines.append(line)
            continue

        if line.strip() == "**Image Text:**":
            out_lines.append(line)
            continue

        # Table rows
        if line.count("|") >= 2 and not _is_table_alignment_row(line):
            leading_pipe = line.lstrip().startswith("|")
            trailing_pipe = line.rstrip().endswith("|")

            parts = line.split("|")
            start = 1 if leading_pipe else 0
            end = -1 if trailing_pipe else len(parts)
            middle = parts[start:end]

            cell_indices: List[Optional[int]] = []
            for part in middle:
                part = part.strip()
                if not part:
                    cell_indices.append(None)
                else:
                    cell_indices.append(len(segments))
                    segments.append(part)

            out_lines.append(None)
            builders.append((
                len(out_lines) - 1,
                "table",
                {
                    "leading_pipe": leading_pipe,
                    "trailing_pipe": trailing_pipe,
                    "cell_indices": cell_indices
                }
            ))
            continue

        # Heading
        m = re.match(r"^(#{1,6}\s+)(.*)$", line)
        if m:
            out_lines.append(None)
            idx = len(segments)
            segments.append(m.group(2).strip())
            builders.append((len(out_lines) - 1, "prefix", {"prefix": m.group(1), "idx": idx}))
            continue

        # Bullet list
        m = re.match(r"^(\s*[-*+]\s+)(.*)$", line)
        if m:
            out_lines.append(None)
            idx = len(segments)
            segments.append(m.group(2).strip())
            builders.append((len(out_lines) - 1, "prefix", {"prefix": m.group(1), "idx": idx}))
            continue

        # Numbered list
        m = re.match(r"^(\s*\d+\.\s+)(.*)$", line)
        if m:
            out_lines.append(None)
            idx = len(segments)
            segments.append(m.group(2).strip())
            builders.append((len(out_lines) - 1, "prefix", {"prefix": m.group(1), "idx": idx}))
            continue

        # Blockquote
        m = re.match(r"^(\s*>\s?)(.*)$", line)
        if m:
            out_lines.append(None)
            idx = len(segments)
            segments.append(m.group(2).strip())
            builders.append((len(out_lines) - 1, "prefix", {"prefix": m.group(1), "idx": idx}))
            continue

        out_lines.append(None)
        idx = len(segments)
        segments.append(line.strip())
        builders.append((len(out_lines) - 1, "prefix", {"prefix": "", "idx": idx}))

    if segments:
        anonymized = _batch_anonymize_texts(segments, cache=cache)
    else:
        anonymized = []

    for line_idx, kind, data in builders:
        if kind == "prefix":
            out_lines[line_idx] = data["prefix"] + anonymized[data["idx"]]
        elif kind == "table":
            leading_pipe = data["leading_pipe"]
            trailing_pipe = data["trailing_pipe"]
            cell_indices = data["cell_indices"]
            cells: List[str] = []
            for idx in cell_indices:
                if idx is None:
                    cells.append("")
                else:
                    cells.append(anonymized[idx])
            rebuilt = ""
            if leading_pipe:
                rebuilt += "| "
            rebuilt += " | ".join(cells)
            if trailing_pipe:
                if rebuilt and not rebuilt.endswith(" "):
                    rebuilt += " "
                rebuilt += "|"
            out_lines[line_idx] = rebuilt

    return "\n".join([l if l is not None else "" for l in out_lines])


def anonymize_markdown_files(extracted_dir: str):

    if not os.path.exists(extracted_dir):
        raise FileNotFoundError(f"Extracted directory not found: {extracted_dir}")

    anonymized_dir = os.path.join(os.path.dirname(extracted_dir), "anonymized")
    os.makedirs(anonymized_dir, exist_ok=True)

    results = {}
    cache: Dict[str, str] = {}
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if not file.lower().endswith(".md"):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    md_text = f.read()

                anonymized_text = anonymize_markdown(md_text, cache=cache)

                out_path = os.path.join(
                    anonymized_dir,
                    os.path.splitext(file)[0] + ".md"
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(anonymized_text)

                results[file] = out_path

            except Exception as e:
                results[file] = f"Error: {e}"

    return {
        "message": "Anonymization completed and saved.",
        "files_saved": list(results.keys()),
        "sample_preview": list(results.items())[0] if results else None
    }
