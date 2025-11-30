import json
from typing import Any, Dict, List

import jsonschema
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .config import OPENAI_CHAT_MODEL
from .prompts import JD_PARSE_INSTRUCTIONS, RATIONALE_INSTRUCTIONS, BIAS_AUDIT_INSTRUCTIONS


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=temperature)


def parse_json_from_llm(prompt: ChatPromptTemplate, llm: ChatOpenAI, input_data: Dict[str, Any]) -> Dict[str, Any]:
    chain = prompt | llm
    resp = chain.invoke(input_data)
    text = resp.content if hasattr(resp, "content") else str(resp)
    # try direct json
    try:
        return json.loads(text)
    except Exception:
        # fallback: extract json substring
        import re
        m = re.search(r"(\{.*\})", text, re.S)
        if not m:
            raise
        return json.loads(m.group(1))


def jd_json_from_text(jd_text: str) -> Dict[str, Any]:
    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JD_PARSE_INSTRUCTIONS),
            ("user", "{jd_text}"),
        ]
    )
    return parse_json_from_llm(prompt, llm, {"jd_text": jd_text})


RATIONALE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source": {"type": "string"},
                    "score_dimension": {"type": "string"},
                },
                "required": ["text", "source"],
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "action": {"type": "string", "enum": ["Shortlist", "Review", "Escalate"]},
    },
    "required": ["summary", "evidence", "confidence", "action"],
}


def generate_rationale_llm(
    jd_json: Dict[str, Any],
    candidate_json: Dict[str, Any],
    evidence_snippets: List[Dict[str, str]],
) -> Dict[str, Any]:
    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RATIONALE_INSTRUCTIONS),
            ("user", "JD_JSON:\n{jd_json}\n\nCANDIDATE_JSON:\n{candidate_json}\n\nEVIDENCE_SNIPPETS:\n{evidence}"),
        ]
    )
    raw = parse_json_from_llm(
        prompt,
        llm,
        {
            "jd_json": json.dumps(jd_json),
            "candidate_json": json.dumps(candidate_json),
            "evidence": json.dumps(evidence_snippets),
        },
    )
    try:
        jsonschema.validate(instance=raw, schema=RATIONALE_SCHEMA)
        return raw
    except Exception:
        # fallback safe structure
        return {
            "summary": "Rationale generation failed.",
            "evidence": evidence_snippets[:2],
            "confidence": 0.0,
            "action": "Review",
        }


def generate_bias_notes_llm(jd_json: Dict[str, Any], resumes: List[str]) -> str:
    llm = get_llm(temperature=0.2)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BIAS_AUDIT_INSTRUCTIONS),
            ("user", "JD_JSON:\n{jd_json}\n\nRESUME_SNIPPETS:\n{resumes}"),
        ]
    )
    chain = prompt | llm
    resp = chain.invoke(
        {
            "jd_json": json.dumps(jd_json),
            "resumes": json.dumps(resumes)[:6000],
        }
    )
    return resp.content if hasattr(resp, "content") else str(resp)
