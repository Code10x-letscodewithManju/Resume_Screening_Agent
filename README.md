
# 📄 Resume Screening Agent — Agentic AI for Recruiter Productivity

**Streamlit + LangGraph + LangChain + OpenAI + Vector Scoring + LLM Rationales + PDF Reports**

A fully agentic, explainable, recruiter-ready resume screening system.
Upload a **Job Description** and **multiple resumes**, and the agent will:

* Parse JD → structured JSON
* Parse resumes (PDF/DOCX) → clean sections
* Compute multi-factor scores (skills, semantic fit, experience, outcomes, risk)
* Rank candidates with weighted scoring
* Generate evidence-backed LLM rationales
* Run a fairness “blind mode” (PII removed) and calculate rank deltas
* Produce a **downloadable PDF report** for each candidate
* Log each screening run for learning and auditability

**Perfect for hackathons, showcases, HR tooling prototypes, agentic-AI portfolios, or ATS integrations.**

---

# 🚀 Features

## 🧠 Agentic Capabilities (LangGraph)

The agent follows a structured DAG:

```
parse_jd → parse_resumes → score_full → score_blind → rationales_and_log → END
```

### ✔️ Perceive

* JD → strict JSON using an LLM
* Resumes → PDF/DOCX parsing + text cleaning + section extraction

### ✔️ Plan

* Apply configurable weights:

  * Skill, Semantic, Experience, Outcome, Risk penalty
* Decide next actions based on the pipeline and scores

### ✔️ Act

* Rank candidates
* Recommend: **Shortlist**, **Review**, or **Escalate**
* Export detailed **PDF candidate reports**

### ✔️ Reason

* Generate structured LLM rationales with:

  * Summary
  * Evidence snippets from resume text
  * Confidence score
  * Explanation of how scores were derived

### ✔️ Learn

* Log each run to `data/logs/runs.jsonl`
* Logs contain JD, weights, candidates, scores, ranks → trainable later

---

# 📊 Scoring System

Each candidate receives the following scores:

| Dimension           | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| **SkillScore**      | Fuzzy skill match between JD skills & resume text          |
| **SemanticScore**   | Embedding similarity (OpenAI) JD ↔ resume                  |
| **ExperienceScore** | YOE extracted vs JD requirements                           |
| **OutcomeScore**    | Matching resume accomplishments vs JD outcomes             |
| **RiskScore**       | Penalizes buzzwords / vague language                       |
| **JDMatchScore**    | Combined alignment: 0.5 Skill + 0.3 Semantic + 0.2 Outcome |
| **CompositeScore**  | Weighted multi-factor scoring (UI sliders)                 |

---

# 📁 Project Structure

Resume-Screening-Agent/
├─ app/
│  ├─ app.py                         # Streamlit UI entry point
│  ├─ __init__.py                    # makes /app a package
│  └─ Agentic_AI/                    # main agentic engine
│     ├─ __init__.py                 # makes /Agentic_AI a package
│     ├─ config.py                   # paths, env, default weights
│     ├─ schemas.py                  # Typed models: JD, ResumeParsed, Scores, CandidateResult
│     ├─ prompts.py                  # JD parser prompt, rationale prompt, bias audit prompt
│     ├─ llm_utils.py                # LangChain ChatOpenAI + JSON enforcement tools
│     ├─ jd_parser.py                # Converts JD text → JD structured object
│     ├─ resume_parser.py            # PDF/DOCX extraction → ResumeParsed
│     ├─ embedding.py                # OpenAI embeddings + cosine similarity
│     ├─ scoring.py                  # Skill/semantic/outcome/experience/risk scoring
│     ├─ utils.py                    # PII redaction, skill token cleanup, text cleaning
│     ├─ reporting.py                # PDF report generation using ReportLab
│     ├─ storage.py                  # JSONL run logging for agent learning
│     └─ graph.py                    # LangGraph agent: state + nodes + flow definition
│
├─ data/
│  ├─ uploads/                       # uploaded resumes (created automatically)
│  ├─ logs/
│  │   └─ runs.jsonl                 # append-only logs (auto-created)
│  └─ sample_resumes/                # optional demo files
│
├─ .env                              # environment variables (not committed)
├─ requirements.txt                  # Python dependencies
├─ README.md                         # project documentation
└─ .gitignore                        # ignore venv, logs, uploads, .env

---

# 🔧 Installation & Setup

## 1️⃣ Clone the repo

```bash
git clone https://github.com/Code10x-letscodewithManju/Resume_Screening_Agent.git
cd Resume_Screening_Agent
```

---

## 2️⃣ Create & activate virtual environment

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4️⃣ Setup environment variables

Copy `.env.example` → `.env`

**Windows:**

```bash
copy .env.example .env
```

**macOS/Linux:**

```bash
cp .env.example .env
```

Then open `.env` and fill in:

```
PYTHONPATH=./app
OPENAI_API_KEY=sk-xxxx
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```

⚠️ **Do NOT use quotes.**

---

## 5️⃣ Run the application

From project root:

```bash
cd app
streamlit run app.py
```

Go to the printed URL:

```
http://localhost:8501
```

You're ready to screen resumes.

---

# 🧪 Usage Guide

## Step 1 — Paste Job Description

* Put your JD in the sidebar text box
* Agent parses → structured JSON (skills, outcomes, experience, risks)

## Step 2 — Adjust Scoring Weights

The sliders change how composite score is computed.

## Step 3 — Upload Resumes

Supports **PDF** and **DOCX**.
You can upload multiple resumes (5–10 ideal for demo).

## Step 4 — Run Screening Agent

The agent will:

* Parse JD
* Parse resumes
* Score (full mode)
* Score (blind mode)
* Generate rationales
* Log run

## Step 5 — See Results

### ✔️ JD Summary (Parsed JSON)

* Must-have / nice-to-have skills
* Expected outcomes
* Experience range
* Risk flags (bias detection)

### ✔️ Ranking Overview & Statistics

* Total candidates
* Avg composite score
* % meeting all must-haves
* Score distributions

### ✔️ Skill Coverage Heatmap

Shows skill-by-skill coverage across candidates.

### ✔️ Candidate Cards

Each card contains:

* Rank (full + blind)
* CompositeScore
* Skill, Semantic, Experience, Outcome, Risk
* Must-have / nice-to-have matches
* Agent recommendation
* LLM rationale with evidence snippets
* Clean resume snippet
* **Downloadable PDF report**

### ✔️ Fairness: Blind Mode

Rank change when PII removed:

* Positive → dropped after anonymization
* Negative → improved after anonymization

Indicates potential bias.

---

# 📝 PDF Export (ReportLab)

Each candidate has a button:

> **Download candidate report (PDF)**

The PDF includes:

* JD snapshot
* Candidate overview
* Full score table
* Skill coverage
* Agent recommendation
* Rationale summary
* Evidence snippets
* Resume snippets

Perfect for recruiters, hiring panels, and audit trails.

---

# 🗃 Logging & Auditability

Every screening run is saved to:

```
data/logs/runs.jsonl
```

Includes:

* JD JSON
* Scoring weights
* Candidates with scores & ranks
* Timestamps

Can be used for:

* Model evaluation
* UX analytics
* Fine-tuning future models
* Fairness audits

---

# 🧩 Extensibility & Future Enhancements

This architecture supports:

### 🔮 ATS integration (Lever, Greenhouse, Workday)

### 📈 Recruiter feedback loop → LLM fine-tuning

### ⚖️ Advanced fairness scoring (gendered-language detectors)

### 🔍 Hybrid search: BM25 + embeddings

### 📊 PDF batch export (all candidates)

### 🌐 Deploy backend + UI via Docker Compose

---

# 🏁 Why This Project Stands Out

* Real agentic workflow (LangGraph)
* Multi-factor scoring
* Semantic search powered by embeddings
* LLM rationales with JSON schema enforcement
* Fairness checks + rank delta analysis
* PDF report generation (recruiter-grade)
* Clean UI using Streamlit
* Fully modular codebase
* Logging pipeline for auditability

---

# 🏆 Credits & Contact

Built by **Manjunath S**
CSE | AI/ML | LLMs | Agentic Systems

(https://github.com/Code10x-letscodewithManju/Resume_Screening_Agent)
