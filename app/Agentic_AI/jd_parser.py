import re
from .schemas import JD
from .llm_utils import jd_json_from_text


def parse_jd(jd_text: str) -> JD:
    jd_json = jd_json_from_text(jd_text)

    jd_json.setdefault("must_have_skills", [])
    jd_json.setdefault("nice_to_have_skills", [])
    jd_json.setdefault("locations", [])
    jd_json.setdefault("employment_type", "unspecified")
    jd_json.setdefault("key_outcomes", [])
    jd_json.setdefault("risk_flags", [])

    if "min_years_experience" not in jd_json or jd_json["min_years_experience"] is None:
        m = re.search(r"(\d+)\+?\s+years?", jd_text, re.I)
        jd_json["min_years_experience"] = float(m.group(1)) if m else 0.0

    if "max_years_experience" not in jd_json or jd_json["max_years_experience"] is None:
        jd_json["max_years_experience"] = jd_json["min_years_experience"] + 5

    return JD(
        role_title=jd_json.get("role_title", "Unspecified Role"),
        must_have_skills=jd_json["must_have_skills"],
        nice_to_have_skills=jd_json["nice_to_have_skills"],
        min_years_experience=float(jd_json["min_years_experience"]),
        max_years_experience=float(jd_json["max_years_experience"]),
        locations=jd_json["locations"],
        employment_type=jd_json["employment_type"],
        key_outcomes=jd_json["key_outcomes"],
        risk_flags=jd_json["risk_flags"],
    )
