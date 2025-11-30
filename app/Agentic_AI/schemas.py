from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class JD:
    role_title: str
    must_have_skills: List[str] = field(default_factory=list)
    nice_to_have_skills: List[str] = field(default_factory=list)
    min_years_experience: float = 0.0
    max_years_experience: float = 0.0
    locations: List[str] = field(default_factory=list)
    employment_type: str = "unspecified"
    key_outcomes: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)


@dataclass
class ResumeParsed:
    resume_id: str
    name: str
    email: Optional[str]
    phone: Optional[str]
    raw_text: str
    sections: Dict[str, str]


@dataclass
class CandidateScores:
    skill_score: float
    semantic_score: float
    experience_score: float
    outcome_score: float
    risk_score: float
    composite_score: float
    # dynamic extra attrs:
    # must_have_hits: List[str]
    # must_have_miss: List[str]
    # nice_to_have_hits: List[str]
    # jd_match_score: float
    # years_experience: float


@dataclass
class CandidateResult:
    resume: ResumeParsed
    scores: CandidateScores
    jd: JD
    rationale: Optional[Dict[str, Any]] = None
    rank_full: Optional[int] = None
    rank_blind: Optional[int] = None
