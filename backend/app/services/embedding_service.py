import oci
import logging
import numpy as np
import re
import time
import random
import os
from typing import List, Tuple

from oci.generative_ai_inference import GenerativeAiInferenceClient
from app.services.chunk_service import token_len
from app.services.secure_config import require_env, get_env
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode
)

logger = logging.getLogger(__name__)

EMBED_MODEL_ID = require_env("EMBED_MODEL_ID")
EXPECTED_VECTOR_DIM = int(get_env("VECTOR_DIM", "1536"))
EMBED_BATCH_MAX_INPUTS = int(get_env("EMBED_BATCH_MAX_INPUTS", "90"))
EMBED_SAFE_MAX_TOKENS = int(get_env("EMBED_SAFE_MAX_TOKENS", "350"))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _split_text_for_embedding(text: str) -> Tuple[str, str]:
    value = _normalize_text(text)
    if not value:
        return "", ""

    midpoint = len(value) // 2
    split_index = value.rfind(" ", 0, midpoint)
    if split_index <= 0:
        split_index = value.find(" ", midpoint)
    if split_index <= 0:
        return value[:midpoint].strip(), value[midpoint:].strip()
    return value[:split_index].strip(), value[split_index:].strip()


class OCIEmbeddingService:

    def __init__(self):
        logger.info("Loading OCI embedding config profile")

        config_profile = require_env("CONFIG_PROFILE")
        oci_config_path = get_env("OCI_CONFIG_PATH", os.path.expanduser("~/.oci/config"))
        self.config = oci.config.from_file(oci_config_path, profile_name=config_profile)
        self.endpoint = require_env("ENDPOINT")
        self.config_profile = config_profile
        self.oci_config_path = oci_config_path

        self.client = GenerativeAiInferenceClient(
            config=self.config,
            service_endpoint=self.endpoint,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240),
        )

        self.serving_mode = OnDemandServingMode(
            model_id=EMBED_MODEL_ID
        )

        self.compartment_id = (
            require_env("COMPARTMENT_ID")
        )

    # PUBLIC FUNCTION
    def embed_text(self, text: str, return_depth=False):
        """
        Embeds text and returns vector.
        If return_depth=True ➝ returns (embedding, depth)
        """
        if not text or not text.strip():
            logger.warning("Empty text received for embedding.")
            return ([], 0) if return_depth else []

        text = _normalize_text(text)
        emb, depth = self._embed_recursive(text)

        if return_depth:
            return emb, depth
        return emb

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embeds a list of texts. Returns a list of vectors in the same order.
        Falls back to per-item embedding if batch fails.
        """
        cleaned: List[str] = []
        for t in texts:
            if not t or not t.strip():
                cleaned.append("")
            else:
                cleaned.append(_normalize_text(t))

        vectors: List[List[float]] = [[] for _ in cleaned]
        batch_texts: List[str] = []
        batch_indices: List[int] = []

        # Pre-filter likely oversized texts to avoid batch failure
        for i, t in enumerate(cleaned):
            if not t:
                vectors[i] = []
                continue
            token_count = token_len(t)
            if token_count > EMBED_SAFE_MAX_TOKENS:
                emb, _ = self._embed_recursive(t)
                vectors[i] = emb
            else:
                batch_texts.append(t)
                batch_indices.append(i)

        if not batch_texts:
            return vectors

        # Try batch embed for remaining using safe sub-batches
        try:
            for start in range(0, len(batch_texts), EMBED_BATCH_MAX_INPUTS):
                sub_batch_texts = batch_texts[start : start + EMBED_BATCH_MAX_INPUTS]
                sub_batch_indices = batch_indices[start : start + EMBED_BATCH_MAX_INPUTS]
                batch_vectors = self._embed_batch_with_retry(sub_batch_texts)
                for idx, emb in zip(sub_batch_indices, batch_vectors):
                    vectors[idx] = emb
            return vectors
        except Exception as e:
            logger.warning(f"Batch embed failed, falling back to per-item: {e}")
            for idx in batch_indices:
                t = cleaned[idx]
                if not t:
                    vectors[idx] = []
                    continue
                emb, _ = self._embed_recursive(t)
                vectors[idx] = emb
            return vectors

    # INTERNAL: Recursive embedding with auto-splitting
    def _embed_recursive(self, text: str, depth: int = 0):
        """
        Returns: (embedding_vector, final_depth_used)
        """
        try:
            req = EmbedTextDetails(
                inputs=[text],
                serving_mode=self.serving_mode,
                compartment_id=self.compartment_id,
            )
            resp = self._embed_call_with_retry(req)
            embedding = resp.data.embeddings[0]
            self._validate_embedding_dimensions(embedding)
            return embedding, depth

        except Exception as e:
            msg = str(e)

            # Not a token-length error → fail
            if "too long" not in msg and "Max tokens" not in msg:
                logger.error(f"Embedding failed at depth={depth}: {e}")
                return [], depth

            # Too long → split
            logger.warning(f"Text too long → splitting at depth={depth}")

            part1, part2 = _split_text_for_embedding(text)

            if not part1 or not part2:
                logger.error("Cannot split further — too small")
                return [], depth

            emb1, d1 = self._embed_recursive(part1, depth + 1)
            emb2, d2 = self._embed_recursive(part2, depth + 1)

            valid = [e for e in [emb1, emb2] if isinstance(e, list) and len(e) > 0]

            if not valid:
                return [], max(d1, d2)

            merged = np.mean(np.array(valid), axis=0).tolist()
            final_depth = max(d1, d2)

            return merged, final_depth

    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        req = EmbedTextDetails(
            inputs=texts,
            serving_mode=self.serving_mode,
            compartment_id=self.compartment_id,
        )
        resp = self._embed_call_with_retry(req)
        embeddings = resp.data.embeddings
        for embedding in embeddings:
            self._validate_embedding_dimensions(embedding)
        return embeddings

    def _validate_embedding_dimensions(self, embedding: List[float]) -> None:
        if embedding and len(embedding) != EXPECTED_VECTOR_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EXPECTED_VECTOR_DIM}, got {len(embedding)}. "
                f"Check EMBED_MODEL_ID and VECTOR_DIM configuration."
            )

    def _embed_call_with_retry(self, req: EmbedTextDetails, max_retries: int = 8):
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                return self.client.embed_text(req)
            except Exception as e:
                last_err = e
                if not self._is_rate_limit_error(e) or attempt == max_retries:
                    raise
                self._backoff_sleep(attempt)
        raise last_err

    def _is_rate_limit_error(self, err: Exception) -> bool:
        if hasattr(err, "status") and err.status == 429:
            return True
        text = str(err)
        return "status': 429" in text or "code': '429" in text or "429" in text

    def _backoff_sleep(self, attempt: int) -> None:
        base = min(60.0, (2 ** attempt))
        jitter = random.uniform(0, 0.5)
        time.sleep(base + jitter)
