import re

PII_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PII_PHONE = re.compile(r"\+?\d[\d\s\-]{8,}")


def redact_pii(text: str) -> str:
    """
    Simple PII redaction for blind mode:
    - Replace emails with [EMAIL]
    - Replace phone numbers with [PHONE]
    - Replace the first line (often the name) with [NAME]
    """
    red = PII_EMAIL.sub("[EMAIL]", text)
    red = PII_PHONE.sub("[PHONE]", red)
    lines = red.splitlines()
    if lines:
        lines[0] = "[NAME]"
    return "\n".join(lines)
