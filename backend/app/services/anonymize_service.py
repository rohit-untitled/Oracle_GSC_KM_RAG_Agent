import os
from app.services.document_loader import load_docx_files
from app.services.docx_extractor import extract_text_with_formatting_in_sequence
from app.services.rag_service import ai_redact_sensitive_info


def anonymize_documents(base_dir: str):
    """
    Loads DOCX files → extracts text → anonymizes text using OCI AI →
    saves anonymized .txt files → returns summary.
    """

    docs = load_docx_files(base_dir)

    anonymized_dir = os.path.join(os.path.dirname(base_dir), "anonymized")
    os.makedirs(anonymized_dir, exist_ok=True)

    result = {}

    for doc in docs:
        file_name = os.path.basename(doc["file_path"])
        try:
            # extract text
            text = extract_text_with_formatting_in_sequence(doc["file_path"])

            # anonymize using OCI AI
            anonymized_text = ai_redact_sensitive_info(text)

            # save anonymized text
            save_path = os.path.join(anonymized_dir, file_name + ".txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(anonymized_text)

        except Exception as e:
            anonymized_text = f"Error: {e}"

        result[file_name] = anonymized_text

    return {
        "message": "Anonymization completed and saved.",
        "files_saved": list(result.keys()),
        "sample_preview": list(result.items())[0]
    }
