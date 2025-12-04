import os
from typing import List
from docx import Document


def load_docx_files(folder_path: str) -> List[dict]:
    """
    Recursively loads all .docx files and returns structured info:
    {
        "file_path": ".../Purchasing/Subfolder/File.docx",
        "folder": "Purchasing/Subfolder",
        "file_name": "File.docx",
        "text": "<extracted text>"
    }
    """
    documents = []
    root_path = os.path.abspath(folder_path)

    for root, _, files in os.walk(folder_path):
        for file in files:

            if not file.lower().endswith(".docx"):
                continue

            # Skip MS Office temporary files
            if file.startswith("~$"):
                continue

            file_path = os.path.join(root, file)
            file_name = os.path.basename(file)

            # Compute relative folder path for clarity
            relative_folder = os.path.relpath(root, root_path)
            if relative_folder == ".":
                relative_folder = os.path.basename(root_path)

            try:
                doc = Document(file_path)
                full_text = "\n".join([p.text for p in doc.paragraphs])

                # OR — if you want formatted output use this:
                # full_text = extract_text_with_formatting_in_sequence(file_path)

                documents.append({
                    "file_path": file_path,
                    "file_name": file_name,
                    "folder": relative_folder,
                    "text": full_text
                })

                print(f"Loaded: {file_path}")

            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

    return documents
