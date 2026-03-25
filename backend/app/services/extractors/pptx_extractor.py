import os
import re
import shutil
import tempfile

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from app.services.extractors.image_summary_service import summarize_image_with_llm


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _save_pptx_image_blob(blob: bytes, ext: str, image_dir: str, img_index: int) -> str:
    ext = ext if ext.startswith(".") else f".{ext}"
    img_name = f"img_{img_index}{ext or '.png'}"
    img_path = os.path.join(image_dir, img_name)
    with open(img_path, "wb") as f:
        f.write(blob)
    return img_path


def extract_pptx_with_formatting_in_sequence(pptx_path: str, image_dir: str | None = None) -> str:
    prs = Presentation(pptx_path)
    formatted_output = []
    temp_dir = image_dir or tempfile.mkdtemp(prefix="pptx_extract_")

    if image_dir:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

    img_counter = 0
    try:
        for slide_no, slide in enumerate(prs.slides, start=1):
            formatted_output.append(f"\n## Slide {slide_no}\n")

            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        image = shape.image
                        img_path = _save_pptx_image_blob(image.blob, image.ext or "png", temp_dir, img_counter)
                        img_counter += 1
                        summary_text = summarize_image_with_llm(img_path)
                        if summary_text:
                            formatted_output.append(f"\n### Image\n\n> {_normalize_text(str(summary_text))}\n")
                    except Exception:
                        continue
                    continue

                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        cells = [cell.text_frame.text.strip().replace("\n", " ") for cell in row.cells]
                        formatted_output.append("| " + " | ".join(cells) + " |")
                    formatted_output.append("\n---\n")
                    continue

                if not shape.has_text_frame:
                    continue

                for paragraph in shape.text_frame.paragraphs:
                    text = (paragraph.text or "").strip()
                    if not text:
                        continue

                    if paragraph.level and paragraph.level > 0:
                        formatted_output.append(f"- {text}")
                    else:
                        formatted_output.append(text)

        final_text = "\n".join(formatted_output)
        return re.sub(r"\n{3,}", "\n\n", final_text)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)