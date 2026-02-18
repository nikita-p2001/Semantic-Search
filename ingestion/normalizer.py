# ingestion/normalizer.py
import re

def normalize_text(text: str) -> str:
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove weird unicode chars (optional)
    text = text.replace("\u200b", "")

    return text.strip()
