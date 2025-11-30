




import pandas as pd
import streamlit as st
from typing import List

from Agentic_AI.config import DATA_DIR, UPLOAD_DIR, DEFAULT_WEIGHTS
from Agentic_AI.graph import build_agent_graph, AgentState
from Agentic_AI.schemas import CandidateResult, JD, ResumeParsed
from Agentic_AI.reporting import build_candidate_report_pdf  # PDF report builder


st.set_page_config(page_title="Resume Screening Agent", layout="wide")

if "graph" not in st.session_state:
    st.session_state["graph"] = build_agent_graph()
graph = st.session_state["graph"]


def save_uploaded_files(files) -> List[str]:
    paths = []
    for f in files:
        dest = UPLOAD_DIR / f.name
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        paths.append(str(dest))
    return paths


st.title("Resume Screening Agent 👩‍💼🤖")

st.markdown(
    """
This tool is an **Agentic AI Resume Screener** powered by LangChain + LangGraph + OpenAI.

**Agent flow:**

1. **Perceive** – parses the Job Description and resumes  
2. **Plan** – applies weighted scoring on skills, semantic fit, experience, outcomes, risk  
3. **Act** – ranks candidates, suggests Shortlist / Review / Escalate  
4. **Reason** – explains decisions with evidence  
5. **Log & Analyze** – logs each run for audit and fairness analysis
"""
)

# Sidebar
with st.sidebar:
    st.header("Step 1 · Paste Job Description")
    jd_text = st.text_area("Job Description", height=260)

    st.header("Step 2 · Configure Scoring")
    skill_w = st.slider("Skill weight", 0.0, 1.0, DEFAULT_WEIGHTS["skill"], 0.05)
    semantic_w = st.slider(
        "Semantic weight", 0.0, 1.0, DEFAULT_WEIGHTS["semantic"], 0.05
    )
    exp_w = st.slider(
        "Experience weight", 0.0, 1.0, DEFAULT_WEIGHTS["experience"], 0.05
    )
    outcome_w = st.slider(
        "Outcome weight", 0.0, 1.0, DEFAULT_WEIGHTS["outcome"], 0.05
    )
    risk_w = st.slider(
        "Risk penalty weight", 0.0, 1.0, DEFAULT_WEIGHTS["risk"], 0.05
    )

    weights = {
        "skill": skill_w,
        "semantic": semantic_w,
        "experience": exp_w,
        "outcome": outcome_w,
        "risk": risk_w,
    }

st.header("Step 3 · Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF/DOCX). For the demo, 3–10 resumes is ideal.",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

col_run, col_bias = st.columns([1, 1])
run_clicked = col_run.button("Run Screening Agent")
bias_clicked = col_bias.button("Show Bias & Fairness Insights (after a run)")

if run_clicked:
    if not jd_text.strip():
        st.error("Please paste a Job Description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Agent perceiving: parsing JD and resumes..."):
            paths = save_uploaded_files(uploaded_files)
            # run LangGraph pipeline until rationales and log
            initial_state: AgentState = {
                "jd_text": jd_text,
                "resume_paths": paths,
                "weights": weights,
            }
            final_state: AgentState = graph.invoke(initial_state)

        jd: JD = final_state["jd"]  # type: ignore
        full_results: List[CandidateResult] = final_state["full_results"]  # type: ignore
        blind_results: List[CandidateResult] = final_state["blind_results"]  # type: ignore

        blind_rank_map = {c.resume.resume_id: c.rank_blind for c in blind_results}
        for c in full_results:
            c.rank_blind = blind_rank_map.get(c.resume.resume_id)

        # JD summary
        st.header("Agent View of the Role (JD Summary)")
        jd_col1, jd_col2 = st.columns([2, 2])
        with jd_col1:
            st.subheader(jd.role_title)
            st.write(
                f"Experience: {jd.min_years_experience}–{jd.max_years_experience} years"
            )
            if jd.locations:
                st.write(f"Preferred locations: {', '.join(jd.locations)}")
            st.write(f"Employment type: {jd.employment_type}")
            st.markdown("**Must-have skills:**")
            st.write(
                ", ".join(jd.must_have_skills) if jd.must_have_skills else "Not detected"
            )
            st.markdown("**Nice-to-have skills:**")
            st.write(
                ", ".join(jd.nice_to_have_skills)
                if jd.nice_to_have_skills
                else "Not detected"
            )

        with jd_col2:
            st.markdown("**Key outcomes expected:**")
            if jd.key_outcomes:
                for o in jd.key_outcomes:
                    st.markdown(f"- {o}")
            else:
                st.write("Not explicitly specified.")
            st.markdown(
                "**Risk flags from JD (potential bias / unrealistic asks):**"
            )
            if jd.risk_flags:
                for rf in jd.risk_flags:
                    st.warning(rf)
            else:
                st.info("No obvious risk flags detected.")

        # Build DataFrame for stats
        st.header("Step 4 · Ranking Overview & Statistics")
        rows = []
        for c in full_results:
            s = c.scores
            must_hits = getattr(s, "must_have_hits", [])
            must_miss = getattr(s, "must_have_miss", [])
            rows.append(
                {
                    "resume_id": c.resume.resume_id,
                    "name": c.resume.name,
                    "Rank (full)": c.rank_full,
                    "Rank (blind)": c.rank_blind,
                    "CompositeScore": s.composite_score,
                    "JDMatchScore": getattr(s, "jd_match_score", 0.0),
                    "SkillScore": s.skill_score,
                    "SemanticScore": s.semantic_score,
                    "ExperienceScore": s.experience_score,
                    "OutcomeScore": s.outcome_score,
                    "RiskScore": s.risk_score,
                    "YearsExp": getattr(s, "years_experience", 0.0),
                    "MustHaveMet": len(must_hits),
                    "MustHaveTotal": len(must_hits) + len(must_miss),
                }
            )
        df = pd.DataFrame(rows)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total candidates", len(df))
        with m2:
            st.metric("Avg composite score", f"{df['CompositeScore'].mean():.3f}")
        with m3:
            if (df["MustHaveTotal"] > 0).any():
                pct_meet_all = (
                    (df["MustHaveMet"] == df["MustHaveTotal"]).mean() * 100
                )
            else:
                pct_meet_all = 0.0
            st.metric("% meeting all must-haves", f"{pct_meet_all:.1f}%")
        with m4:
            st.metric("Avg JDMatchScore", f"{df['JDMatchScore'].mean():.3f}")

        st.subheader("Ranked Candidates (table view)")
        st.dataframe(
            df.sort_values("Rank (full)").reset_index(drop=True),
            use_container_width=True,
        )

        st.subheader("Score Distributions")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.bar_chart(df[["CompositeScore"]])
        with sc2:
            st.bar_chart(df[["SkillScore", "SemanticScore", "ExperienceScore"]])

        st.subheader("Fairness: rank change in blind mode")
        if df["Rank (blind)"].notna().all():
            df["RankDelta"] = df["Rank (blind)"] - df["Rank (full)"]
            st.bar_chart(df[["RankDelta"]])
            st.caption(
                "Positive RankDelta = candidate moved down when PII removed; negative = moved up."
            )

        # Candidate cards
        st.header("Step 5 · Candidate Cards (Reasoning, Actions & Reports)")
        for c in full_results:
            s = c.scores
            must_hits = getattr(s, "must_have_hits", [])
            must_miss = getattr(s, "must_have_miss", [])
            nice_hits = getattr(s, "nice_to_have_hits", [])
            jd_match = getattr(s, "jd_match_score", 0.0)
            years = getattr(s, "years_experience", 0.0)

            with st.container():
                st.markdown("---")
                left, right = st.columns([1.5, 2])

                with left:
                    st.markdown(f"### {c.resume.name}")
                    st.caption(f"Resume ID: {c.resume.resume_id}")
                    st.write(f"Email: {c.resume.email or 'N/A'}")
                    st.write(f"Phone: {c.resume.phone or 'N/A'}")

                    st.metric("Rank (full)", c.rank_full)
                    if c.rank_blind is not None:
                        delta = c.rank_blind - (c.rank_full or 0)
                        st.metric("Rank (blind)", c.rank_blind, delta=delta)

                    st.write(f"Composite Score: **{s.composite_score:.3f}**")
                    st.write(f"JDMatchScore: **{jd_match:.3f}**")
                    st.write(
                        f"Skill: {s.skill_score:.3f} | Semantic: {s.semantic_score:.3f} | "
                        f"Exp: {s.experience_score:.3f} | Outcome: {s.outcome_score:.3f} | "
                        f"Risk: {s.risk_score:.3f}"
                    )
                    st.write(f"Estimated years of experience: {years:.1f}")

                    st.markdown("**Must-have skills coverage:**")
                    total_must = len(must_hits) + len(must_miss)
                    st.write(f"Met: {len(must_hits)} / {total_must}")
                    if must_hits:
                        st.caption("Matched: " + ", ".join(must_hits))
                    if must_miss:
                        st.caption("Missing: " + ", ".join(must_miss))

                    if nice_hits:
                        st.markdown("**Nice-to-have skills matched:**")
                        st.caption(", ".join(nice_hits))

                    if c.rationale:
                        st.markdown("**Agent Recommendation:**")
                        st.write(f"Action: **{c.rationale.get('action', 'Review')}**")
                        st.write(f"Confidence: {c.rationale.get('confidence', 0.0):.2f}")

                    # —— PDF export button ——
                    pdf_bytes = build_candidate_report_pdf(c, jd)
                    safe_name = c.resume.name.replace(" ", "_") or "candidate"
                    st.download_button(
                        label="Download candidate report (PDF)",
                        data=pdf_bytes,
                        file_name=f"{safe_name}_report.pdf",
                        mime="application/pdf",
                    )

                with right:
                    st.markdown("**Agent Rationale & Evidence**")
                    if c.rationale:
                        st.write(c.rationale.get("summary", ""))
                        for ev in c.rationale.get("evidence", []):
                            with st.expander(
                                f"{ev.get('score_dimension', 'dimension')} · {ev.get('source', 'source')}"
                            ):
                                st.write(ev.get("text", "")[:1200])
                    else:
                        st.info("No rationale generated for this candidate.")

                    st.markdown("**Resume Snippets**")
                    if "summary" in c.resume.sections:
                        with st.expander("Summary section"):
                            st.write(c.resume.sections["summary"][:1000])
                    if "experience" in c.resume.sections:
                        with st.expander("Experience section"):
                            st.write(c.resume.sections["experience"][:1000])
                    if "skills" in c.resume.sections:
                        with st.expander("Skills section"):
                            st.write(c.resume.sections["skills"][:1000])

        st.success("Run logged to data/logs/runs.jsonl.")

if bias_clicked:
    st.info(
        "Run the screening first; fairness uses the parsed JD and resumes from the last run."
    )
