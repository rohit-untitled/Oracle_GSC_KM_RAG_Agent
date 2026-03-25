import base64
import logging
import mimetypes
import os

from oci_openai import OciOpenAI, OciUserPrincipalAuth
from PIL import Image

from app.services.secure_config import get_env, require_env

logger = logging.getLogger(__name__)

_OCI_CLIENT = None
_OCI_REGION = require_env("OCI_REGION")
_OCI_PROFILE = require_env("CONFIG_PROFILE")
_OCI_COMPARTMENT_ID = require_env("COMPARTMENT_ID")
_OCI_MODEL = require_env("OCI_OPENAI_MODEL")
IMAGE_SUMMARY_ENABLED = (get_env("IMAGE_SUMMARY_ENABLED", "true") or "true").strip().lower() in {"1", "true", "yes", "y"}


def _get_oci_client() -> OciOpenAI:
    global _OCI_CLIENT
    if _OCI_CLIENT is None:
        _OCI_CLIENT = OciOpenAI(
            region=_OCI_REGION,
            auth=OciUserPrincipalAuth(profile_name=_OCI_PROFILE),
            compartment_id=_OCI_COMPARTMENT_ID,
        )
    return _OCI_CLIENT


def _detect_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type:
        return mime_type

    try:
        with Image.open(image_path) as img:
            fmt = (img.format or "").lower()
        mime_map = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
            "webp": "image/webp",
        }
        return mime_map.get(fmt, "image/png")
    except Exception:
        return "image/png"


def summarize_image_with_llm(image_path: str) -> str:
    if not IMAGE_SUMMARY_ENABLED:
        return ""

    if not os.path.exists(image_path):
        return ""

    mime_type = _detect_mime_type(image_path)

    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        logger.warning("Failed to read image %s: %s", image_path, e)
        return ""

    try:
        client = _get_oci_client()
        completion = client.chat.completions.create(
            model=_OCI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an expert multimodal document analyst operating in an enterprise environment "
                                "to support a production-grade Retrieval-Augmented Generation (RAG) system.\n\n"
                                "Task:\n"
                                "Analyze the provided image and extract only factual technical or business information explicitly present.\n"
                                "Do not hallucinate, infer, or include decorative details.\n"
                                "If the image has no meaningful technical/business content, return EMPTY."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        payload = completion.model_dump()
        return payload["choices"][0]["message"]["content"] or ""
    except Exception as e:
        logger.warning("LLM image summary failed for %s: %s", image_path, e)
        return ""