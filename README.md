# 🎯 HR Matching Agent

An AI-powered HR candidate matching system built with **LangGraph**, **Groq (LLaMA 3.3)**, and **RAG (ChromaDB + HuggingFace Embeddings)**. The agent automates resume screening, candidate ranking, and interview preparation through a conversational Streamlit interface.

> **Milestone 3 Submission** — LangGraph Agent Architecture

---

## 📌 Project Overview

This agent takes a job description, searches through a resume database using semantic search, ranks candidates, and generates detailed hiring reports — all through natural language conversation. Users can refine requirements mid-conversation and the agent re-ranks accordingly.

---

## 🏗️ Agent Architecture

The agent is built as a **LangGraph state machine** with the following workflow:

```
START
  ↓
Parse JD          → reads job description file
  ↓
Extract Req.      → splits must-have vs nice-to-have skills
  ↓
Search Resumes    → RAG semantic search across all resumes
  ↓
Rank Candidates   → scores and compares shortlisted candidates
  ↓
Generate Report   → detailed match report + interview questions
  ↓
Human Feedback    → user can refine, re-rank, or approve
  ↓
END
```

### Agent State Tracks:
- Full conversation history
- Parsed job requirements (must-have / nice-to-have)
- Candidate shortlist with scores and reasoning
- Current workflow stage and screening round
- User refinement history

---

## 📁 Project Structure

```
hr_matching_agent/
├── matching_agent.py        # Core LangGraph agent + state machine
├── tools.py                 # All 10 LangChain tools
├── app.py                   # Streamlit chat interface
├── test_scenarios.py        # 6 conversation flow tests
├── data/
│   ├── job_descriptions/    # Add JD .txt files here
│   │   └── senior_fullstack_dev.txt
│   └── resumes/             # Add resume .txt/.pdf files here
│       ├── john_chen.txt
│       ├── priya_sharma.txt
│       ├── alex_morgan.txt
│       ├── rahul_verma.txt
│       └── sarah_johnson.txt
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph 0.2+ |
| LLM | Groq — LLaMA 3.3 70B Versatile (free) |
| Embeddings | HuggingFace — all-MiniLM-L6-v2 (free) |
| Vector Store | ChromaDB |
| Chat Interface | Streamlit |
| Document Loading | LangChain Community |
| PDF Support | PyPDF |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Rexter9/hr_matching_agent.git
cd hr_matching_agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install sentence-transformers langchain-text-splitters langchain-groq
```

### 3. Set up API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_key_here
```
Get your **free** Groq API key at: https://console.groq.com

### 4. Build the resume index
```bash
python -c "from tools import rebuild_resume_index; print(rebuild_resume_index.invoke({}))"
```
Expected output: `Index rebuilt successfully with X chunks from resume files.`

### 5. Launch the app
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🧰 Tools Implemented (Part A)

| Tool | Description |
|------|-------------|
| `list_resume_files` | List all available resume files |
| `read_resume` | Read full content of a resume |
| `list_jd_files` | List job description files |
| `read_job_description` | Read a job description |
| `rag_search_resumes` | Semantic search across all resumes |
| `rebuild_resume_index` | Rebuild ChromaDB vector index |
| `extract_requirements` | Parse JD into must-have / nice-to-have |
| `compare_candidates` | Head-to-head candidate comparison |
| `generate_interview_questions` | Tailored screening questions per candidate |
| `generate_match_report` | Full match report with hire recommendation |

---

## 💬 Conversational Features (Part B)

The agent understands natural language queries like:

```
"Find me candidates with React and 3+ years experience"
"Compare the top 3 matches side by side"
"Why did John rank higher than Priya?"
"Re-rank prioritizing AWS experience over Python"
"Generate interview questions for the top candidate"
```

Users can **refine requirements mid-conversation** and the agent re-ranks with full explanation of what changed and why.

---

## 🔁 Multi-Round Screening (Part C)

| Round | Description |
|-------|-------------|
| Round 1 | Initial screen — top candidates from full resume pool |
| Round 2 | Deep analysis — detailed match reports for top 3 |
| Round 3 | Final decision — hire / hold / reject with full justification |

---

## 🧪 Test Scenarios

Run all 6 conversation flow tests:
```bash
python test_scenarios.py        # all scenarios
python test_scenarios.py 1      # specific scenario
```

| # | Scenario | Tests |
|---|----------|-------|
| 1 | Full Workflow | End-to-end JD → report |
| 2 | Natural Language Search | Skill-based candidate search |
| 3 | Iterative Refinement | Re-ranking as requirements change |
| 4 | Head-to-Head Comparison | Side-by-side candidate comparison |
| 5 | Explainability | Why did X rank higher than Y? |
| 6 | Multi-Round Screening | 3-round elimination process |

---

## 📊 Sample Candidates (Included)

| Candidate | Experience | Strength | Weakness |
|-----------|-----------|----------|----------|
| John Chen | 7 years | React + Python expert, AWS, mentoring | — (best match) |
| Priya Sharma | 6 years | Python/AWS expert, IIT grad | React only 1 year |
| Rahul Verma | 5 years | Solid full-stack breadth | No TypeScript |
| Sarah Johnson | 8 years | AWS expert, strong system design | React/Python only 8 months |
| Alex Morgan | 4 years | React/TypeScript expert | Below 5yr requirement, weak backend |

---

## 📋 Submission Checklist

- [x] LangGraph-based agent (`matching_agent.py`)
- [x] State machine diagram (see Architecture section above)
- [x] Streamlit chat interface (`app.py`)
- [x] 6 test conversation flows (`test_scenarios.py`)
- [ ] Demo video (5–6 min) — record `streamlit run app.py` walkthrough

---

## 🔒 Security Note

The `.env` file containing your API key is listed in `.gitignore` and will **never** be committed to this repository.

---

## 👤 Author

**Faiz** — https://github.com/Rexter9