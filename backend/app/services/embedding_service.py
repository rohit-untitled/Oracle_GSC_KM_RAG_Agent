import oci
import logging
import numpy as np
import re

from oci.generative_ai_inference import GenerativeAiInferenceClient
from app.services.secure_config import require_env
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode
)

logger = logging.getLogger(__name__)


class OCIEmbeddingService:

    def __init__(self):
        logger.info("Loading OCI config profile ")

        self.config = oci.config.from_file("~/.oci/config", profile_name=require_env("CONFIG_PROFILE"))
        # self.endpoint = "https://inference.generativeai.ap-hyderabad-1.oci.oraclecloud.com"
        self.endpoint = require_env("ENDPOINT")

        logger.info("Initializing OCI Generative AI Client...")
        self.client = GenerativeAiInferenceClient(
            config=self.config,
            service_endpoint=self.endpoint,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240),
        )

        self.serving_mode = OnDemandServingMode(
            model_id="cohere.embed-multilingual-image-v3.0"
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

        text = re.sub(r"\s+", " ", text).strip()
        emb, depth = self._embed_recursive(text)

        if return_depth:
            return emb, depth
        return emb

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
            resp = self.client.embed_text(req)
            return resp.data.embeddings[0], depth

        except Exception as e:
            msg = str(e)

            # Not a token-length error → fail
            if "too long" not in msg and "Max tokens" not in msg:
                logger.error(f"Embedding failed at depth={depth}: {e}")
                return [], depth

            # Too long → split
            logger.warning(f"Text too long → splitting at depth={depth}")

            words = text.split()
            mid = len(words) // 2

            if mid == 0:
                logger.error("Cannot split further — too small")
                return [], depth

            part1 = " ".join(words[:mid])
            part2 = " ".join(words[mid:])

            emb1, d1 = self._embed_recursive(part1, depth + 1)
            emb2, d2 = self._embed_recursive(part2, depth + 1)

            valid = [e for e in [emb1, emb2] if isinstance(e, list) and len(e) > 0]

            if not valid:
                return [], max(d1, d2)

            merged = np.mean(np.array(valid), axis=0).tolist()
            final_depth = max(d1, d2)

            return merged, final_depth
