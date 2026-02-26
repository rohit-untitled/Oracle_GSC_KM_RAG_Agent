import os
import re
import json
import time
import random
import logging
from typing import Dict, List, Optional, Tuple
from oci_openai import OciOpenAI, OciUserPrincipalAuth
from app.services.secure_config import require_env

logger = logging.getLogger(__name__)

_OCI_CLIENT = None
_OCI_REGION = require_env("OCI_REGION")
_OCI_PROFILE = require_env("CONFIG_PROFILE")
_OCI_COMPARTMENT_ID = require_env("COMPARTMENT_ID")
_OCI_MODEL = require_env("OCI_OPENAI_MODEL")
ANON_MAX_RETRIES = int(os.getenv("ANON_MAX_RETRIES", "5"))
ANON_MAX_TOKENS = int(os.getenv("ANON_MAX_TOKENS", "4000"))
ANON_TEMPERATURE = float(os.getenv("ANON_TEMPERATURE", "0"))


def _get_oci_client() -> OciOpenAI:
    global _OCI_CLIENT
    if _OCI_CLIENT is None:
        _OCI_CLIENT = OciOpenAI(
            region=_OCI_REGION,
            auth=OciUserPrincipalAuth(profile_name=_OCI_PROFILE),
            compartment_id=_OCI_COMPARTMENT_ID,
        )
    return _OCI_CLIENT


def _chat_with_retry(call_fn, message: str, max_retries: int = ANON_MAX_RETRIES) -> str:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return call_fn(message)
        except Exception as e:
            last_err = e
            if not _is_rate_limit_error(e) or attempt == max_retries:
                raise
            _backoff_sleep(attempt)
    raise last_err


def _is_rate_limit_error(err: Exception) -> bool:
    if hasattr(err, "status") and err.status == 429:
        return True
    if hasattr(err, "code") and str(getattr(err, "code")) == "429":
        return True
    text = str(err)
    lowered = text.lower()
    return (
        "status': 429" in text
        or '"status": 429' in text
        or "code': '429'" in text
        or '"code": "429"' in text
        or "too many requests" in lowered
        or "rate limit" in lowered
    )


def _backoff_sleep(attempt: int) -> None:
    base = min(60.0, (2 ** attempt))
    jitter = random.uniform(0, 0.5)
    time.sleep(base + jitter)


def _extract_completion_text(completion) -> str:
    payload = completion.model_dump() if hasattr(completion, "model_dump") else {}
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if txt:
                    parts.append(txt)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def ai_redact_sensitive_info(text: str) -> str:
    user_message = f"""
You are a data anonymization system. Your job is to redact sensitive info while preserving the original
format, structure, and wording as much as possible.

Rules:
- Replace all company, customer, client, partner, vendor, and organization names that are NOT "Oracle"
  (or obvious Oracle variants like "Oracle Cloud", "Oracle OCI") with [Anonymized Customer].
- Replace personal names, emails, phone numbers, account numbers, IDs, URLs, and IPs with [Anonymized].
- Do NOT change "Oracle" or its obvious variants.
- Do NOT add commentary, explanations, or extra text.
- Preserve punctuation, line breaks, markdown, tables, bullets, and headings.

Return ONLY the anonymized text.

Original Text:
{text}

Anonymized Text:
"""

    def _call_chat(message: str) -> str:
        client = _get_oci_client()
        completion = client.chat.completions.create(
            model=_OCI_MODEL,
            messages=[{"role": "user", "content": message}],
            max_tokens=ANON_MAX_TOKENS,
            temperature=ANON_TEMPERATURE,
        )
        return _extract_completion_text(completion)

    return _chat_with_retry(_call_chat, user_message)


def ai_redact_sensitive_info_batch(texts: List[str]) -> List[str]:
    """
    Batch anonymization. Returns a list of anonymized strings in the same order.
    """
    payload = json.dumps(texts, ensure_ascii=False)
    user_message = f"""
You are a data anonymization system. Anonymize each string in the JSON array below.

Rules:
- Replace all company, customer, client, partner, vendor, and organization names that are NOT "Oracle"
  (or obvious Oracle variants like "Oracle Cloud", "Oracle OCI") with [Anonymized Customer].
- Replace personal names, emails, phone numbers, account numbers, IDs, URLs, and IPs with [Anonymized].
- Do NOT change "Oracle" or its obvious variants.
- Do NOT add commentary, explanations, or extra text.
- Return ONLY a JSON array of strings, same length and order as the input.

Input JSON:
{payload}
"""

    def _call_chat(message: str) -> str:
        client = _get_oci_client()
        completion = client.chat.completions.create(
            model=_OCI_MODEL,
            messages=[{"role": "user", "content": message}],
            max_tokens=ANON_MAX_TOKENS,
            temperature=ANON_TEMPERATURE,
        )
        return _extract_completion_text(completion)

    raw = _chat_with_retry(_call_chat, user_message)
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _is_table_alignment_row(line: str) -> bool:
    stripped = line.strip().strip("|").strip()
    if not stripped:
        return False
    return bool(re.fullmatch(r"[:\-\s]+", stripped))


def _is_output_blocklist_error(err: Exception) -> bool:
    text = str(err).lower()
    return "output blocklist" in text or "inappropriate output content" in text


def _regex_redact_sensitive_info(text: str) -> str:
    """
    Local fallback redaction when LLM anonymization is blocked/unavailable.
    Keeps markdown structure while masking common sensitive patterns.
    """
    out = text
    out = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[Anonymized]", out)
    out = re.sub(r"\b(?:https?://|www\.)\S+\b", "[Anonymized]", out)
    out = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[Anonymized]", out)
    out = re.sub(r"\b\+?\d[\d\-\s()]{7,}\d\b", "[Anonymized]", out)
    out = re.sub(r"\b[A-Z]{2,}\d{2,}[A-Z0-9\-]*\b", "[Anonymized]", out)
    out = re.sub(r"\b\d{9,}\b", "[Anonymized]", out)
    # Conservative redaction for likely names/org labels in fallback mode.
    out = re.sub(
        r"\b(?!Oracle\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b",
        "[Anonymized Customer]",
        out,
    )
    return out


def _safe_blocklist_fallback(text: str) -> str:
    """
    Fail closed for blocked output: preserve minimal structure while preventing leakage.
    """
    base = _regex_redact_sensitive_info(text).strip()
    if not base:
        return ""
    return "[Anonymized]"


def _batch_anonymize_texts(
    texts: List[str],
    cache: Optional[Dict[str, str]] = None,
    max_chars: int = 6000,
    metrics: Optional[Dict[str, int]] = None,
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
        try:
            anonymized = ai_redact_sensitive_info_batch(batch)
            if not isinstance(anonymized, list) or len(anonymized) != len(batch):
                raise ValueError("Invalid anonymization batch response length/type")
        except Exception as e:
            logger.warning("Batch anonymization failed; applying fallback path: %s", e)
            if metrics is not None:
                metrics["batch_failures"] = metrics.get("batch_failures", 0) + 1
            anonymized = []
            for item in batch:
                try:
                    anonymized.append(ai_redact_sensitive_info(item))
                except Exception as e_item:
                    if _is_output_blocklist_error(e_item):
                        logger.warning("Output blocklist hit during anonymization; using regex fallback.")
                        if metrics is not None:
                            metrics["blocklist_fallbacks"] = metrics.get("blocklist_fallbacks", 0) + 1
                        anonymized.append(_safe_blocklist_fallback(item))
                    else:
                        logger.warning("Per-item anonymization failed; using regex fallback: %s", e_item)
                        if metrics is not None:
                            metrics["regex_fallbacks"] = metrics.get("regex_fallbacks", 0) + 1
                        anonymized.append(_regex_redact_sensitive_info(item))
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


def anonymize_markdown(
    md_text: str,
    cache: Optional[Dict[str, str]] = None,
    metrics: Optional[Dict[str, int]] = None,
) -> str:
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
        if metrics is not None:
            metrics["segments_total"] = metrics.get("segments_total", 0) + len(segments)
        anonymized = _batch_anonymize_texts(segments, cache=cache, metrics=metrics)
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
    saved_files: List[str] = []
    metrics: Dict[str, int] = {
        "files_processed": 0,
        "files_saved": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "segments_total": 0,
        "batch_failures": 0,
        "blocklist_fallbacks": 0,
        "regex_fallbacks": 0,
    }
    cache: Dict[str, str] = {}
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if not file.lower().endswith(".md"):
                continue
            metrics["files_processed"] += 1
            file_path = os.path.join(root, file)
            try:
                rel_root = os.path.relpath(root, extracted_dir)
                target_dir = anonymized_dir if rel_root == "." else os.path.join(anonymized_dir, rel_root)
                os.makedirs(target_dir, exist_ok=True)
                out_path = os.path.join(
                    target_dir,
                    os.path.splitext(file)[0] + ".md"
                )
                if os.path.exists(out_path):
                    results[file] = f"Skipped (already anonymized): {out_path}"
                    metrics["files_skipped"] += 1
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    md_text = f.read()

                anonymized_text = anonymize_markdown(md_text, cache=cache, metrics=metrics)

                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(anonymized_text)

                results[file] = out_path
                saved_files.append(file)
                metrics["files_saved"] += 1

            except Exception as e:
                results[file] = f"Error: {e}"
                metrics["files_failed"] += 1

    return {
        "message": "Anonymization completed and saved.",
        "files_saved": saved_files,
        "file_status": results,
        "metrics": metrics,
        "sample_preview": list(results.items())[0] if results else None
    }
