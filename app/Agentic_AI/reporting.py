import io
from datetime import datetime
from typing import List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .schemas import CandidateResult, JD


def _heading(text: str) -> Paragraph:
    styles = getSampleStyleSheet()
    style = styles["Heading3"]
    return Paragraph(text, style)


def _body(text: str) -> Paragraph:
    styles = getSampleStyleSheet()
    style = styles["BodyText"]
    style.leading = 14
    return Paragraph(text.replace("\n", "<br/>"), style)


def build_candidate_report_pdf(
    candidate: CandidateResult,
    jd: JD,
) -> bytes:
    """
    Build a single-candidate report PDF and return it as bytes.
    Intended to be used with Streamlit st.download_button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
        title=f"Candidate Report - {candidate.resume.name}",
    )

    story: List = []

    # ---- Title ----
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title = Paragraph(f"Candidate Report: {candidate.resume.name}", title_style)
    story.append(title)
    story.append(Spacer(1, 12))

    subtitle = Paragraph(
        f"Role: {jd.role_title} &nbsp;&nbsp;|&nbsp;&nbsp; Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        styles["Normal"],
    )
    story.append(subtitle)
    story.append(Spacer(1, 16))

    # ---- JD snapshot ----
    story.append(_heading("Job Description Snapshot"))
    jd_text = []
    jd_text.append(f"<b>Must-have skills:</b> {', '.join(jd.must_have_skills) or 'N/A'}")
    jd_text.append(f"<b>Nice-to-have skills:</b> {', '.join(jd.nice_to_have_skills) or 'N/A'}")
    jd_text.append(
        f"<b>Experience range:</b> {jd.min_years_experience}–{jd.max_years_experience} years"
    )
    if jd.key_outcomes:
        jd_text.append(f"<b>Key outcomes:</b> {', '.join(jd.key_outcomes)}")
    if jd.risk_flags:
        jd_text.append(f"<b>Risk flags:</b> {', '.join(jd.risk_flags)}")

    story.append(_body("<br/>".join(jd_text)))
    story.append(Spacer(1, 16))

    # ---- Candidate overview ----
    s = candidate.scores
    story.append(_heading("Candidate Overview"))

    overview_lines = [
        f"<b>Name:</b> {candidate.resume.name}",
        f"<b>Resume ID:</b> {candidate.resume.resume_id}",
        f"<b>Email:</b> {candidate.resume.email or 'N/A'}",
        f"<b>Phone:</b> {candidate.resume.phone or 'N/A'}",
        f"<b>Rank (full):</b> {candidate.rank_full or '-'}",
        f"<b>Rank (blind):</b> {candidate.rank_blind or '-'}",
        f"<b>CompositeScore:</b> {s.composite_score:.3f}",
        f"<b>JDMatchScore:</b> {getattr(s, 'jd_match_score', 0.0):.3f}",
        f"<b>Estimated years of experience:</b> {getattr(s, 'years_experience', 0.0):.1f}",
    ]
    story.append(_body("<br/>".join(overview_lines)))
    story.append(Spacer(1, 16))

    # ---- Score table ----
    story.append(_heading("Score Breakdown"))
    data = [
        ["Dimension", "Score"],
        ["SkillScore", f"{s.skill_score:.3f}"],
        ["SemanticScore", f"{s.semantic_score:.3f}"],
        ["ExperienceScore", f"{s.experience_score:.3f}"],
        ["OutcomeScore", f"{s.outcome_score:.3f}"],
        ["RiskScore", f"{s.risk_score:.3f}"],
        ["CompositeScore", f"{s.composite_score:.3f}"],
        ["JDMatchScore", f"{getattr(s, 'jd_match_score', 0.0):.3f}"],
    ]
    table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 16))

    # ---- Skill coverage ----
    story.append(_heading("Skill Coverage"))
    must_hits = getattr(s, "must_have_hits", [])
    must_miss = getattr(s, "must_have_miss", [])
    nice_hits = getattr(s, "nice_to_have_hits", [])

    skill_lines = []
    skill_lines.append(
        f"<b>Must-have skills met:</b> {len(must_hits)} / {len(must_hits) + len(must_miss)}"
    )
    if must_hits:
        skill_lines.append(f"<b>Matched:</b> {', '.join(must_hits)}")
    if must_miss:
        skill_lines.append(f"<b>Missing:</b> {', '.join(must_miss)}")
    if nice_hits:
        skill_lines.append(f"<b>Nice-to-have skills matched:</b> {', '.join(nice_hits)}")

    story.append(_body("<br/>".join(skill_lines)))
    story.append(Spacer(1, 16))

    # ---- Agent recommendation & rationale ----
    story.append(_heading("Agent Recommendation"))

    if candidate.rationale:
        r = candidate.rationale
        rec_lines = [
            f"<b>Action:</b> {r.get('action', 'Review')}",
            f"<b>Confidence:</b> {r.get('confidence', 0.0):.2f}",
        ]
        story.append(_body("<br/>".join(rec_lines)))
        story.append(Spacer(1, 8))

        story.append(_heading("Summary"))
        story.append(_body(r.get("summary", "")))
        story.append(Spacer(1, 8))

        evs = r.get("evidence", [])
        if evs:
            story.append(_heading("Evidence Snippets"))
            for ev in evs:
                text = ev.get("text", "")
                src = ev.get("source", "")
                dim = ev.get("score_dimension", "")
                para = _body(
                    f"<b>Source:</b> {src} &nbsp;&nbsp; <b>Dimension:</b> {dim}<br/>{text}"
                )
                story.append(para)
                story.append(Spacer(1, 6))
    else:
        story.append(_body("No rationale generated for this candidate."))

    story.append(Spacer(1, 16))

    # ---- Resume snippets (optional, trimmed) ----
    story.append(_heading("Resume Snippets"))

    if "summary" in candidate.resume.sections:
        story.append(_body(f"<b>Summary section:</b><br/>{candidate.resume.sections['summary'][:1000]}"))
        story.append(Spacer(1, 8))
    if "experience" in candidate.resume.sections:
        story.append(_body(f"<b>Experience section:</b><br/>{candidate.resume.sections['experience'][:1200]}"))
        story.append(Spacer(1, 8))
    if "skills" in candidate.resume.sections:
        story.append(_body(f"<b>Skills section:</b><br/>{candidate.resume.sections['skills'][:800]}"))
        story.append(Spacer(1, 8))

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
