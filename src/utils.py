import re
from typing import Iterable


def clean_text(raw_text: str) -> str:
    """Normalize resume text while preserving resume-relevant structure."""
    if raw_text is None:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)

    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        cleaned_lines.append(re.sub(r"\s{2,}", " ", stripped))

    final_text = "\n".join(cleaned_lines).strip()
    final_text = re.sub(r"\*\*+", "", final_text)
    final_text = re.sub(r"\u00a0", " ", final_text)
    return final_text


def clean_jd_text(raw_text: str) -> str:
    """Use the same cleaning logic for job descriptions without losing key skills or experience details."""
    return clean_text(raw_text)


def normalize_list(values: Iterable[str]) -> list[str]:
    """Return a cleaned list of non-empty strings."""
    normalized = []
    for value in values:
        candidate = str(value).strip()
        if candidate:
            normalized.append(candidate)
    return normalized
