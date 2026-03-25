import re


def extract_txt_in_sequence(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return re.sub(r"\n{3,}", "\n\n", text).strip()