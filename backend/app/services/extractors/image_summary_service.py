import base64
import logging
import mimetypes
import os

from oci_openai import OciOpenAI, OciUserPrincipalAuth
from PIL import Image

from app.services.secure_config import get_env, require_env

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/webp",
}

UNSUPPORTED_IMAGE_EXTENSIONS = {
    ".emf",
    ".wmf",
    ".svg",
    ".ico",
}

MAX_IMAGE_BYTES = int(get_env("IMAGE_SUMMARY_MAX_BYTES", str(10 * 1024 * 1024)))

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
    if mime_type in SUPPORTED_IMAGE_MIME_TYPES:
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
        return mime_map.get(fmt, "")
    except Exception:
        return ""


def _should_skip_image(image_path: str, mime_type: str) -> bool:
    ext = os.path.splitext(image_path)[1].lower()
    if ext in UNSUPPORTED_IMAGE_EXTENSIONS:
        logger.info("Skipping unsupported image extension for LLM summary: %s", image_path)
        return True

    if not mime_type:
        logger.info("Skipping image with unknown/unsupported MIME type for LLM summary: %s", image_path)
        return True

    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        logger.info("Skipping unsupported MIME type %s for LLM summary: %s", mime_type, image_path)
        return True

    try:
        file_size = os.path.getsize(image_path)
    except OSError as e:
        logger.warning("Failed to inspect image size %s: %s", image_path, e)
        return True

    if file_size > MAX_IMAGE_BYTES:
        logger.info(
            "Skipping oversized image for LLM summary (%s bytes > %s): %s",
            file_size,
            MAX_IMAGE_BYTES,
            image_path,
        )
        return True

    return False


def summarize_image_with_llm(image_path: str) -> str:
    if not IMAGE_SUMMARY_ENABLED or not os.path.exists(image_path):
        return ""

    mime_type = _detect_mime_type(image_path)
    if _should_skip_image(image_path, mime_type):
        return ""

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
                                "Analyze the provided image and extract only factual technical or business information explicitly present. "
                                "Do not hallucinate or include decorative details. If there is no meaningful technical/business content, return EMPTY."
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