# from typing import List, Dict
# import re

# from .schemas import JD, ResumeParsed, CandidateScores, CandidateResult
# from .embedding import embed_texts, cosine_similarity
# from .utils import redact_pii


# def _extract_years(text: str) -> float:
#     m = re.search(r"(\d+)\+?\s+years?", text, re.I)
#     if not m:
#         return 0.0
#     return float(m.group(1))


# def compute_scores(
#     jd: JD,
#     resume: ResumeParsed,
#     weights: Dict[str, float],
#     jd_embed_vec,
#     resume_embed_vec,
# ) -> CandidateScores:
#     text_lower = resume.raw_text.lower()

#     must = [s.strip() for s in jd.must_have_skills if s.strip()]
#     must_hits = [s for s in must if s.lower() in text_lower]
#     must_miss = [s for s in must if s.lower() not in text_lower]
#     skill_score = len(must_hits) / max(len(must), 1)

#     nice = [s.strip() for s in jd.nice_to_have_skills if s.strip()]
#     nice_hits = [s for s in nice if s.lower() in text_lower]
#     nice_score = len(nice_hits) / max(len(nice), 1) if nice else 0.0

#     semantic_score = cosine_similarity(jd_embed_vec, resume_embed_vec)

#     exp_section = resume.sections.get("experience", resume.raw_text)
#     years = _extract_years(exp_section)
#     if jd.min_years_experience > 0:
#         experience_score = min(years / jd.min_years_experience, 1.0)
#     else:
#         experience_score = 0.5

#     outcome_hits = 0
#     for o in jd.key_outcomes:
#         chunk = o.lower()[:20]
#         if chunk and chunk in text_lower:
#             outcome_hits += 1
#     outcome_score = (
#         outcome_hits / max(len(jd.key_outcomes), 1) if jd.key_outcomes else 0.0
#     )

#     buzzwords = ["hard-working", "team player", "self-starter", "passionate"]
#     buzz = sum(1 for b in buzzwords if b in text_lower)
#     has_metrics = bool(re.search(r"\d+%", text_lower)) or bool(re.search(r"\d{4}", text_lower))
#     risk_score = 0.2
#     if buzz > 2 and not has_metrics:
#         risk_score = 0.7

#     jd_match_score = 0.5 * skill_score + 0.3 * semantic_score + 0.2 * outcome_score

#     composite = (
#         weights.get("skill", 0.4) * skill_score +
#         weights.get("semantic", 0.3) * semantic_score +
#         weights.get("experience", 0.15) * experience_score +
#         weights.get("outcome", 0.1) * outcome_score -
#         weights.get("risk", 0.05) * risk_score
#     )

#     scores = CandidateScores(
#         skill_score=float(skill_score),
#         semantic_score=float(semantic_score),
#         experience_score=float(experience_score),
#         outcome_score=float(outcome_score),
#         risk_score=float(risk_score),
#         composite_score=float(composite),
#     )

#     scores.must_have_hits = must_hits              # type: ignore[attr-defined]
#     scores.must_have_miss = must_miss              # type: ignore[attr-defined]
#     scores.nice_to_have_hits = nice_hits           # type: ignore[attr-defined]
#     scores.jd_match_score = float(jd_match_score)  # type: ignore[attr-defined]
#     scores.years_experience = float(years)         # type: ignore[attr-defined]

#     return scores


# def rank_candidates(
#     jd: JD,
#     resumes: List[ResumeParsed],
#     weights: Dict[str, float],
#     blind_mode: bool = False,
# ) -> List[CandidateResult]:
#     if blind_mode:
#         texts = [redact_pii(r.raw_text) for r in resumes]
#     else:
#         texts = [r.raw_text for r in resumes]

#     jd_text_for_embed = " ".join(
#         [jd.role_title] + jd.must_have_skills + jd.nice_to_have_skills + jd.key_outcomes
#     )
#     jd_embed = embed_texts([jd_text_for_embed])[0]
#     resume_embeds = embed_texts(texts)

#     results: List[CandidateResult] = []
#     for idx, r in enumerate(resumes):
#         scores = compute_scores(
#             jd=jd,
#             resume=r,
#             weights=weights,
#             jd_embed_vec=jd_embed,
#             resume_embed_vec=resume_embeds[idx],
#         )
#         results.append(
#             CandidateResult(
#                 resume=r,
#                 scores=scores,
#                 jd=jd,
#             )
#         )

#     results_sorted = sorted(results, key=lambda c: c.scores.composite_score, reverse=True)
#     return results_sorted
















from typing import List, Dict
import re

from .schemas import JD, ResumeParsed, CandidateScores, CandidateResult
from .embedding import embed_texts, cosine_similarity
from .utils import redact_pii


# --- Helpers for better skill matching ---

WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase tokenization (letters/digits only)."""
    return set(WORD_RE.findall(text.lower()))


def _skill_matches(skill: str, resume: ResumeParsed) -> bool:
    """
    Decide if a JD skill is "present" in the resume using token overlap.

    - Use tokens from the whole resume + the skills section (if present).
    - For short skills (1–2 tokens): at least 1 token must appear.
    - For medium skills (3–4 tokens): at least 50% of tokens must appear.
    - For long skills (5+ tokens): at least 40% of tokens must appear.
    """
    skill_tokens = _tokenize(skill)
    if not skill_tokens:
        return False

    # Tokens from entire resume
    all_tokens = _tokenize(resume.raw_text)

    # Boost: explicitly include tokens from the "skills" section if it exists
    skills_section = resume.sections.get("skills", "")
    if skills_section:
        all_tokens |= _tokenize(skills_section)

    overlap = skill_tokens & all_tokens
    if not overlap:
        return False

    overlap_ratio = len(overlap) / len(skill_tokens)

    n_tokens = len(skill_tokens)
    if n_tokens <= 2:
        # at least 1 token present
        return len(overlap) >= 1
    elif n_tokens <= 4:
        # at least half the tokens present
        return overlap_ratio >= 0.5
    else:
        # long sentence-like skills; allow partial match
        return overlap_ratio >= 0.4


def _extract_years(text: str) -> float:
    m = re.search(r"(\d+)\+?\s+years?", text, re.I)
    if not m:
        return 0.0
    return float(m.group(1))


def compute_scores(
    jd: JD,
    resume: ResumeParsed,
    weights: Dict[str, float],
    jd_embed_vec,
    resume_embed_vec,
) -> CandidateScores:
    # --- Skill coverage (must-have & nice-to-have) ---
    must = [s.strip() for s in jd.must_have_skills if s.strip()]
    must_hits = [s for s in must if _skill_matches(s, resume)]
    must_miss = [s for s in must if not _skill_matches(s, resume)]
    skill_score = len(must_hits) / max(len(must), 1)

    nice = [s.strip() for s in jd.nice_to_have_skills if s.strip()]
    nice_hits = [s for s in nice if _skill_matches(s, resume)]
    nice_score = len(nice_hits) / max(len(nice), 1) if nice else 0.0

    # --- Semantic similarity (JD vs resume) ---
    semantic_score = cosine_similarity(jd_embed_vec, resume_embed_vec)

    # --- Experience score ---
    exp_section = resume.sections.get("experience", resume.raw_text)
    years = _extract_years(exp_section)
    if jd.min_years_experience > 0:
        experience_score = min(years / jd.min_years_experience, 1.0)
    else:
        experience_score = 0.5

    # --- Outcome score (JD outcomes language present in resume) ---
    text_lower = resume.raw_text.lower()
    outcome_hits = 0
    for o in jd.key_outcomes:
        chunk = o.lower()[:20]
        if chunk and chunk in text_lower:
            outcome_hits += 1
    outcome_score = (
        outcome_hits / max(len(jd.key_outcomes), 1) if jd.key_outcomes else 0.0
    )

    # --- Risk score (buzzwords without evidence/metrics) ---
    buzzwords = ["hard-working", "team player", "self-starter", "passionate"]
    buzz = sum(1 for b in buzzwords if b in text_lower)
    has_metrics = bool(re.search(r"\d+%", text_lower)) or bool(
        re.search(r"\d{4}", text_lower)
    )
    risk_score = 0.2
    if buzz > 2 and not has_metrics:
        risk_score = 0.7

    # JDMatchScore combines main alignment components
    jd_match_score = 0.5 * skill_score + 0.3 * semantic_score + 0.2 * outcome_score

    # Composite score with recruiter-defined weights
    composite = (
        weights.get("skill", 0.4) * skill_score
        + weights.get("semantic", 0.3) * semantic_score
        + weights.get("experience", 0.15) * experience_score
        + weights.get("outcome", 0.1) * outcome_score
        - weights.get("risk", 0.05) * risk_score
    )

    scores = CandidateScores(
        skill_score=float(skill_score),
        semantic_score=float(semantic_score),
        experience_score=float(experience_score),
        outcome_score=float(outcome_score),
        risk_score=float(risk_score),
        composite_score=float(composite),
    )

    # Attach extra attributes for UI / logging
    scores.must_have_hits = must_hits  # type: ignore[attr-defined]
    scores.must_have_miss = must_miss  # type: ignore[attr-defined]
    scores.nice_to_have_hits = nice_hits  # type: ignore[attr-defined]
    scores.jd_match_score = float(jd_match_score)  # type: ignore[attr-defined]
    scores.years_experience = float(years)  # type: ignore[attr-defined]

    return scores


def rank_candidates(
    jd: JD,
    resumes: List[ResumeParsed],
    weights: Dict[str, float],
    blind_mode: bool = False,
) -> List[CandidateResult]:
    # Optionally redact PII for blind mode
    if blind_mode:
        texts = [redact_pii(r.raw_text) for r in resumes]
    else:
        texts = [r.raw_text for r in resumes]

    # Embed JD and resumes once
    jd_text_for_embed = " ".join(
        [jd.role_title] + jd.must_have_skills + jd.nice_to_have_skills + jd.key_outcomes
    )
    jd_embed = embed_texts([jd_text_for_embed])[0]
    resume_embeds = embed_texts(texts)

    results: List[CandidateResult] = []
    for idx, r in enumerate(resumes):
        scores = compute_scores(
            jd=jd,
            resume=r,
            weights=weights,
            jd_embed_vec=jd_embed,
            resume_embed_vec=resume_embeds[idx],
        )
        results.append(
            CandidateResult(
                resume=r,
                scores=scores,
                jd=jd,
            )
        )

    # Sort by composite score (higher is better)
    results_sorted = sorted(
        results, key=lambda c: c.scores.composite_score, reverse=True
    )
    return results_sorted
