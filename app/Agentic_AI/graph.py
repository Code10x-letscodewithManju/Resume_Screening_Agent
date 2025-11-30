from typing import List, TypedDict, Dict, Any

from langgraph.graph import StateGraph, END

from .schemas import JD, ResumeParsed, CandidateResult
from .config import DEFAULT_WEIGHTS
from .jd_parser import parse_jd
from .resume_parser import parse_resume
from .scoring import rank_candidates
from .llm_utils import generate_rationale_llm, generate_bias_notes_llm
from .storage import log_run


class AgentState(TypedDict, total=False):
    jd_text: str
    jd: JD
    resume_paths: List[str]
    resumes: List[ResumeParsed]
    weights: Dict[str, float]
    full_results: List[CandidateResult]
    blind_results: List[CandidateResult]
    bias_notes: str  # optional, can be filled by node_bias_notes if used separately


def node_parse_jd(state: AgentState) -> AgentState:
    jd = parse_jd(state["jd_text"]) # type: ignore
    return {"jd": jd}


def node_parse_resumes(state: AgentState) -> AgentState:
    resumes = [parse_resume(p) for p in state["resume_paths"]]  # type: ignore
    return {"resumes": resumes}


def node_score_full(state: AgentState) -> AgentState:
    jd = state["jd"]   # type: ignore
    resumes = state["resumes"]  # type: ignore
    weights = state.get("weights", DEFAULT_WEIGHTS)
    full_results = rank_candidates(jd, resumes, weights, blind_mode=False)
    for i, c in enumerate(full_results):
        c.rank_full = i + 1
    return {"full_results": full_results}


def node_score_blind(state: AgentState) -> AgentState:
    jd = state["jd"]    # type: ignore
    resumes = state["resumes"]      # type: ignore
    weights = state.get("weights", DEFAULT_WEIGHTS)
    blind_results = rank_candidates(jd, resumes, weights, blind_mode=True)
    for i, c in enumerate(blind_results):
        c.rank_blind = i + 1
    return {"blind_results": blind_results}


def node_rationales_and_log(state: AgentState) -> AgentState:
    jd = state["jd"]    # type: ignore
    full_results = state["full_results"]    # type: ignore

    jd_json = {
        "role_title": jd.role_title,
        "must_have_skills": jd.must_have_skills,
        "nice_to_have_skills": jd.nice_to_have_skills,
        "min_years_experience": jd.min_years_experience,
        "max_years_experience": jd.max_years_experience,
        "key_outcomes": jd.key_outcomes,
        "risk_flags": jd.risk_flags,
    }

    # Generate rationales for top-K candidates
    for i, c in enumerate(full_results):
        if i >= 3:
            break
        evidence = []
        if "skills" in c.resume.sections:
            evidence.append(
                {
                    "text": c.resume.sections["skills"][:600],
                    "source": "resume.skills",
                    "score_dimension": "SkillScore",
                }
            )
        if "experience" in c.resume.sections:
            evidence.append(
                {
                    "text": c.resume.sections["experience"][:600],
                    "source": "resume.experience",
                    "score_dimension": "ExperienceScore",
                }
            )
        candidate_json = {
            "resume_id": c.resume.resume_id,
            "name": c.resume.name,
            "scores": {
                "CompositeScore": c.scores.composite_score,
                "SkillScore": c.scores.skill_score,
                "SemanticScore": c.scores.semantic_score,
                "ExperienceScore": c.scores.experience_score,
                "OutcomeScore": c.scores.outcome_score,
                "RiskScore": c.scores.risk_score,
            },
        }
        c.rationale = generate_rationale_llm(jd_json, candidate_json, evidence)

    # Prepare log entry
    serializable_candidates = []
    for c in full_results:
        s = c.scores
        serializable_candidates.append(
            {
                "resume_id": c.resume.resume_id,
                "name": c.resume.name,
                "rank_full": c.rank_full,
                "scores": {
                    "CompositeScore": s.composite_score,
                    "JDMatchScore": getattr(s, "jd_match_score", 0.0),
                    "SkillScore": s.skill_score,
                    "SemanticScore": s.semantic_score,
                    "ExperienceScore": s.experience_score,
                    "OutcomeScore": s.outcome_score,
                    "RiskScore": s.risk_score,
                    "YearsExp": getattr(s, "years_experience", 0.0),
                },
            }
        )

    log_run(jd_json, state.get("weights", DEFAULT_WEIGHTS), serializable_candidates)
    # We mutated full_results in place; return state unchanged
    return state


def node_bias_notes(state: AgentState) -> AgentState:
    """
    Optional node – NOT wired into the main graph to avoid name collision.
    You can call this manually if you want a fairness narrative.
    """
    jd = state["jd"]        # type: ignore
    resumes = state["resumes"]      # type: ignore
    jd_json = {
        "role_title": jd.role_title,
        "must_have_skills": jd.must_have_skills,
        "nice_to_have_skills": jd.nice_to_have_skills,
        "min_years_experience": jd.min_years_experience,
        "max_years_experience": jd.max_years_experience,
        "risk_flags": jd.risk_flags,
    }
    texts = [r.raw_text[:2000] for r in resumes]
    notes = generate_bias_notes_llm(jd_json, texts)
    return {"bias_notes": notes}


def build_agent_graph():
    graph = StateGraph(AgentState)

    # Main pipeline nodes
    graph.add_node("parse_jd", node_parse_jd)
    graph.add_node("parse_resumes", node_parse_resumes)
    graph.add_node("score_full", node_score_full)
    graph.add_node("score_blind", node_score_blind)
    graph.add_node("rationales_and_log", node_rationales_and_log)

    # Entry and edges for main flow
    graph.set_entry_point("parse_jd")
    graph.add_edge("parse_jd", "parse_resumes")
    graph.add_edge("parse_resumes", "score_full")
    graph.add_edge("score_full", "score_blind")
    graph.add_edge("score_blind", "rationales_and_log")
    graph.add_edge("rationales_and_log", END)

    return graph.compile()
