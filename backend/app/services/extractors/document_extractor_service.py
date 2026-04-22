import os

from app.services.docx_extractor import extract_docx_with_formatting_in_sequence
from app.services.extractors.excel_extractor import extract_excel_workbook, workbook_record_to_text
from app.services.extractors.pdf_extractor import extract_pdf_with_formatting_in_sequence
from app.services.extractors.pptx_extractor import extract_pptx_with_formatting_in_sequence
from app.services.extractors.txt_extractor import extract_txt_in_sequence


def extract_text_with_formatting_in_sequence(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".docx":
        return extract_docx_with_formatting_in_sequence(file_path)
    if ext == ".pptx":
        return extract_pptx_with_formatting_in_sequence(file_path)
    if ext == ".txt":
        return extract_txt_in_sequence(file_path)
    if ext == ".pdf":
        return extract_pdf_with_formatting_in_sequence(file_path)
    if ext in {".xlsx", ".xlsm"}:
        return workbook_record_to_text(extract_excel_workbook(file_path))

    raise ValueError(f"Unsupported file type: {ext}. Supported: .docx, .pptx, .txt, .pdf, .xlsx, .xlsm")