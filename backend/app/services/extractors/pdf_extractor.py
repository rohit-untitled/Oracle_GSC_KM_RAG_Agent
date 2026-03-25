import os
import re
import shutil
import tempfile

import pypdfium2 as pdfium

from app.services.extractors.image_summary_service import summarize_image_with_llm


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def extract_pdf_with_formatting_in_sequence(pdf_path: str, image_dir: str | None = None) -> str:
    pdf = pdfium.PdfDocument(pdf_path)
    formatted_output = []
    temp_dir = image_dir or tempfile.mkdtemp(prefix="pdf_extract_")

    if image_dir:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

    try:
        for page_no in range(len(pdf)):
            page = pdf[page_no]
            formatted_output.append(f"\n## Page {page_no + 1}\n")

            page_text = ""
            try:
                text_page = page.get_textpage()
                page_text = (text_page.get_text_bounded() or "").strip()
                text_page.close()
            except Exception:
                page_text = ""

            if page_text:
                formatted_output.append(page_text)
            else:
                try:
                    bitmap = page.render(scale=2)
                    pil_image = bitmap.to_pil()
                    img_path = os.path.join(temp_dir, f"pdf_page_{page_no + 1}.png")
                    pil_image.save(img_path)
                    summary_text = summarize_image_with_llm(img_path)
                    if summary_text:
                        formatted_output.append(f"\n### Image\n\n> {_normalize_text(str(summary_text))}\n")
                except Exception:
                    pass

            page.close()

        final_text = "\n".join(formatted_output)
        return re.sub(r"\n{3,}", "\n\n", final_text)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)