from pathlib import Path
from typing import Dict, Optional
import uuid
import re

import pdfplumber
import docx2txt

from .schemas import ResumeParsed

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s\-]{8,}")


def _extract_text_from_pdf(path: Path) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text += t + "\n"
    return text


def _extract_text_from_docx(path: Path) -> str:
    return docx2txt.process(str(path)) or ""


def _detect_sections(text: str) -> Dict[str, str]:
    lower = text.lower()
    sections = {"raw": text}
    markers = ["summary", "skills", "experience", "education", "projects"]
    for m in markers:
        idx = lower.find(m)
        if idx != -1:
            sections[m] = text[idx: idx + 2000]
    return sections


def _extract_name(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return "Unknown Candidate"


def _extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None


def _extract_phone(text: str) -> Optional[str]:
    m = PHONE_RE.search(text)
    return m.group(0) if m else None


def parse_resume(file_path: str) -> ResumeParsed:
    p = Path(file_path)
    if p.suffix.lower() == ".pdf":
        text = _extract_text_from_pdf(p)
    elif p.suffix.lower() in [".docx", ".doc"]:
        text = _extract_text_from_docx(p)
    else:
        text = p.read_text(encoding="utf-8", errors="ignore")

    sections = _detect_sections(text)
    name = _extract_name(text)
    email = _extract_email(text)
    phone = _extract_phone(text)

    return ResumeParsed(
        resume_id=str(uuid.uuid4()),
        name=name,
        email=email,
        phone=phone,
        raw_text=text,
        sections=sections,
    )
