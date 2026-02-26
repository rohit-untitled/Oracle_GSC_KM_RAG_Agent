import base64
import mimetypes
import os
import re
import shutil

import docx
import pypdfium2 as pdfium
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from oci_openai import OciOpenAI, OciSessionAuth, OciUserPrincipalAuth
from PIL import Image
from tqdm import tqdm
from app.services.secure_config import require_env

_OCI_CLIENT = None
_OCI_REGION = require_env("OCI_REGION")
_OCI_PROFILE = require_env("CONFIG_PROFILE")
_OCI_COMPARTMENT_ID = require_env("COMPARTMENT_ID")
_OCI_MODEL = require_env("OCI_OPENAI_MODEL")


def _get_oci_client() -> OciOpenAI:
    global _OCI_CLIENT
    if _OCI_CLIENT is None:
        _OCI_CLIENT = OciOpenAI(
            region=_OCI_REGION,
            auth=OciUserPrincipalAuth(profile_name=_OCI_PROFILE),
            compartment_id=_OCI_COMPARTMENT_ID,
        )
    return _OCI_CLIENT


def _summarize_image_with_llm(image_path: str) -> str:
    if not os.path.exists(image_path):
        return ""

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
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
            mime_type = mime_map.get(fmt, "image/png")
        except Exception:
            mime_type = "image/png"

    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        print(f"Failed to read image {image_path}: {e}")
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
                                "Analyze the provided image exactly as a senior functional or technical stakeholder would. "
                                "Extract ONLY the factual technical and/or business information explicitly present in the image, "
                                "such that the diagram or content can be fully understood without seeing the image.\n\n"

                                "Strict Extraction Rules:\n"
                                "- Extract information EXACTLY as shown; preserve original terminology and wording.\n"
                                "- Never hallucinate, infer, assume, or add missing details.\n"
                                "- Do NOT infer intent, purpose, or meaning beyond what is visually present.\n"
                                "- Ignore and exclude all non-functional or decorative content, including:\n"
                                "  * Headers, footers, cover pages, logos, branding elements\n"
                                "  * Icons, symbols, or images with no technical or business meaning\n"
                                "- Do NOT include any visual or stylistic details such as colors, fonts, layout, shapes, "
                                "alignment, textures, or design aesthetics.\n\n"

                                "Extraction Requirements:\n"
                                "- First, perform a literal extraction of all technical and business information present.\n"
                                "- Explicitly describe relationships, flows, sequences, dependencies, or hierarchies if shown.\n"
                                "- Omit any fields, sections, or concepts not explicitly present in the image.\n\n"

                                "Output Constraints:\n"
                                "- If the image contains only visual elements (e.g., logos or decorative graphics) and no "
                                "technical or business information, return an EMPTY response.\n"
                                "- Do NOT output negative filler such as 'no content found' or similar statements.\n\n"

                                "Final Output:\n"
                                "1. Provide a concise, structured extraction of the information that exists.\n"
                                "2. End with a detailed technical summary explaining the diagram, blocks, or flows and "
                                "how they relate to each other, written so the reader can fully understand the content "
                                "without seeing the image."
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
        print(f"LLM image summary failed for {image_path}: {e}")
        return ""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _save_image_part(img_part, image_dir: str, img_index: int) -> str:
    partname = str(img_part.partname)
    ext = os.path.splitext(partname)[1] if partname else ""
    if not ext:
        ext = ".png"
    img_name = f"img_{img_index}{ext}"
    img_path = os.path.join(image_dir, img_name)
    with open(img_path, "wb") as f:
        f.write(img_part.blob)
    return img_path


def _extract_images_from_paragraphs(paragraphs, rels, image_dir, img_counter):
    image_summaries = []

    for paragraph in paragraphs:
        for run in paragraph.runs:
            blips = run.element.xpath(".//a:blip")
            if not blips:
                continue

            for blip in blips:
                embed_id = blip.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                )

                if not embed_id or embed_id not in rels:
                    continue

                img_part = rels[embed_id].target_part
                img_path = _save_image_part(img_part, image_dir, img_counter[0])
                img_counter[0] += 1

                summary_text = _summarize_image_with_llm(img_path)
                if summary_text:
                    image_summaries.append(_normalize_text(str(summary_text)))

    return image_summaries


def _extract_docx_with_formatting_in_sequence(docx_path, image_dir="temp_docx_images_seq"):
    """
    Extract structured text from DOCX including:
    - Headings (converted to markdown #)
    - Bullet lists
    - Inline images (LLM; optional OCR fallback)
    - Tables
    - Paragraphs in exact sequential order
    """

    doc = docx.Document(docx_path)
    formatted_output = []

    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir, exist_ok=True)

    rels = doc.part.rels
    para_index = 0
    table_index = 0
    img_counter = [0]
    section_index = 0

    for block in tqdm(
        doc.element.body.iterchildren(),
        desc=f"Extracting: {os.path.basename(docx_path)}",
        ncols=100,
    ):
        # 1. PARAGRAPH
        if block.tag.endswith("p"):
            paragraph = doc.paragraphs[para_index]
            para_index += 1

            image_texts = _extract_images_from_paragraphs(
                [paragraph], rels, image_dir, img_counter
            )

            text = paragraph.text.strip()
            if not text and not image_texts:
                continue
            section_index += 1

            style_name = paragraph.style.name.lower()

            # Headings
            if "heading" in style_name:
                lvl = re.findall(r"\d+", style_name)
                lvl = int(lvl[0]) if lvl else 1
                formatted_output.append(f"\n{'#' * lvl} {text}\n")

            # Bulleted list
            elif paragraph._element.xpath(".//w:numPr"):
                formatted_output.append(f"- {text}")

            # Normal paragraph
            else:
                if text:
                    formatted_output.append(text)

            for ocr_text in image_texts:
                formatted_output.append(f"\n### Image\n\n> {ocr_text}\n")

        # 2. TABLE
        elif block.tag.endswith("tbl"):
            section_index += 1

            if table_index < len(doc.tables):
                table = doc.tables[table_index]
            else:
                table = None
            table_index += 1

            if table is not None:
                for row in table.rows:
                    cells = []
                    for cell in row.cells:
                        cell_text = cell.text.strip().replace("\n", " ")
                        cell_image_texts = _extract_images_from_paragraphs(
                            cell.paragraphs, rels, image_dir, img_counter
                        )
                        if cell_image_texts:
                            if cell_text:
                                cell_text += " "
                            cell_text += "Image: " + " / ".join(cell_image_texts)
                        cells.append(cell_text)
                    formatted_output.append("| " + " | ".join(cells) + " |")

            formatted_output.append("\n---\n")

    # Cleanup long empty spaces
    final_text = "\n".join(formatted_output)
    final_text = re.sub(r"\n{3,}", "\n\n", final_text)

    shutil.rmtree(image_dir, ignore_errors=True)

    return final_text


def _save_pptx_image_blob(blob: bytes, ext: str, image_dir: str, img_index: int) -> str:
    ext = ext if ext.startswith(".") else f".{ext}"
    img_name = f"img_{img_index}{ext or '.png'}"
    img_path = os.path.join(image_dir, img_name)
    with open(img_path, "wb") as f:
        f.write(blob)
    return img_path


def _extract_pptx_with_formatting_in_sequence(pptx_path, image_dir="temp_pptx_images_seq"):
    prs = Presentation(pptx_path)
    formatted_output = []

    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir, exist_ok=True)

    img_counter = 0

    for slide_no, slide in enumerate(prs.slides, start=1):
        formatted_output.append(f"\n## Slide {slide_no}\n")

        for shape in slide.shapes:
            # Extract image-only shapes and summarize via multimodal LLM
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                try:
                    image = shape.image
                    img_path = _save_pptx_image_blob(
                        image.blob,
                        image.ext or "png",
                        image_dir,
                        img_counter,
                    )
                    img_counter += 1
                    summary_text = _summarize_image_with_llm(img_path)
                    if summary_text:
                        formatted_output.append(f"\n### Image\n\n> {_normalize_text(str(summary_text))}\n")
                except Exception:
                    continue
                continue

            if shape.has_table:
                table = shape.table
                for row in table.rows:
                    cells = []
                    for cell in row.cells:
                        cells.append(cell.text_frame.text.strip().replace("\n", " "))
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
    final_text = re.sub(r"\n{3,}", "\n\n", final_text)
    shutil.rmtree(image_dir, ignore_errors=True)
    return final_text


def _extract_txt_in_sequence(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_pdf_with_formatting_in_sequence(pdf_path: str, image_dir="temp_pdf_images_seq") -> str:
    pdf = pdfium.PdfDocument(pdf_path)
    formatted_output = []

    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir, exist_ok=True)

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
            # Fallback for scanned/image-only pages.
            try:
                bitmap = page.render(scale=2)
                pil_image = bitmap.to_pil()
                img_path = os.path.join(image_dir, f"pdf_page_{page_no + 1}.png")
                pil_image.save(img_path)
                summary_text = _summarize_image_with_llm(img_path)
                if summary_text:
                    formatted_output.append(f"\n### Image\n\n> {_normalize_text(str(summary_text))}\n")
            except Exception:
                pass

        page.close()

    final_text = "\n".join(formatted_output)
    final_text = re.sub(r"\n{3,}", "\n\n", final_text)
    shutil.rmtree(image_dir, ignore_errors=True)
    return final_text


def extract_text_with_formatting_in_sequence(file_path, image_dir="temp_docx_images_seq"):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".docx":
        return _extract_docx_with_formatting_in_sequence(file_path, image_dir=image_dir)
    if ext == ".pptx":
        pptx_image_dir = image_dir.replace("docx", "pptx")
        return _extract_pptx_with_formatting_in_sequence(file_path, image_dir=pptx_image_dir)
    if ext == ".txt":
        return _extract_txt_in_sequence(file_path)
    if ext == ".pdf":
        pdf_image_dir = image_dir.replace("docx", "pdf")
        return _extract_pdf_with_formatting_in_sequence(file_path, image_dir=pdf_image_dir)

    raise ValueError(f"Unsupported file type: {ext}. Supported: .docx, .pptx, .txt, .pdf")
