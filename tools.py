"""
tools.py — All LangChain tools for the HR Matching Agent
Includes: file system tools, RAG search, requirement extraction,
          candidate comparison, and interview question generation.
"""
import os, json, re, shutil
from pathlib import Path
from typing import List, Optional
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
JD_DIR     = DATA_DIR / "job_descriptions"
RESUME_DIR = DATA_DIR / "resumes"
CHROMA_DIR = "chroma_db"

for d in [JD_DIR, RESUME_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Shared LLM ───────────────────────────────────────────────────────────────

def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ─── Embeddings (HuggingFace — fully free) ────────────────────────────────────

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ─── File System Tools ────────────────────────────────────────────────────────

@tool
def list_resume_files() -> str:
    """List all resume files available in the resumes directory."""
    files = list(RESUME_DIR.glob("*"))
    if not files:
        return "No resume files found. Add .txt or .pdf files to data/resumes/"
    return "\n".join(f.name for f in files)

@tool
def read_resume(filename: str) -> str:
    """Read the full text content of a specific resume file.

    Args:
        filename: Name of the resume file (e.g. 'john_doe.txt')
    """
    path = RESUME_DIR / filename
    if not path.exists():
        return f"File not found: {filename}"
    if path.suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        return "\n".join(d.page_content for d in docs)
    return path.read_text(encoding="utf-8")

@tool
def list_jd_files() -> str:
    """List all job description files available."""
    files = list(JD_DIR.glob("*"))
    if not files:
        return "No JD files found. Add .txt files to data/job_descriptions/"
    return "\n".join(f.name for f in files)

@tool
def read_job_description(filename: str) -> str:
    """Read the full text of a job description file.

    Args:
        filename: Name of the JD file (e.g. 'senior_engineer.txt')
    """
    path = JD_DIR / filename
    if not path.exists():
        files = list(JD_DIR.glob("*.txt"))
        if files:
            return files[0].read_text(encoding="utf-8")
        return f"File not found: {filename}"
    return path.read_text(encoding="utf-8")

# ─── RAG / Vector Store ───────────────────────────────────────────────────────

def build_vector_store() -> Chroma:
    """Load resumes into Chroma vector store (run once or when files change)."""
    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    docs = []
    for f in RESUME_DIR.glob("*"):
        try:
            if f.suffix == ".pdf":
                loader = PyPDFLoader(str(f))
            else:
                loader = TextLoader(str(f), encoding="utf-8")
            raw = loader.load()
            chunks = splitter.split_documents(raw)
            for chunk in chunks:
                chunk.metadata["source_file"] = f.name
                chunk.metadata["candidate_id"] = f.stem
            docs.extend(chunks)
        except Exception as e:
            print(f"[WARN] Could not load {f.name}: {e}")

    if not docs:
        print("[WARN] No resume documents loaded for vector store.")
        return Chroma(embedding_function=embeddings, persist_directory=CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        docs, embeddings, persist_directory=CHROMA_DIR
    )
    return vectorstore

def get_vector_store() -> Chroma:
    """Get existing vector store or build it."""
    embeddings = get_embeddings()
    if Path(CHROMA_DIR).exists():
        return Chroma(embedding_function=embeddings, persist_directory=CHROMA_DIR)
    return build_vector_store()

@tool
def rag_search_resumes(query: str, k: int = 5) -> str:
    """Search resumes using semantic similarity via RAG.

    Args:
        query: Natural language search query (e.g. 'React developer 3 years experience')
        k: Number of results to return (default 5)
    """
    try:
        vs = get_vector_store()
        results = vs.similarity_search_with_score(query, k=k)
        if not results:
            return "No matching resumes found. Make sure resumes are indexed."

        output = []
        seen = set()
        for doc, score in results:
            cid = doc.metadata.get("candidate_id", "unknown")
            if cid not in seen:
                seen.add(cid)
                output.append(
                    f"Candidate: {cid} (similarity: {1-score:.2f})\n"
                    f"Excerpt: {doc.page_content[:300]}...\n"
                )
        return "\n---\n".join(output)
    except Exception as e:
        return f"RAG search error: {e}. Run rebuild_resume_index first."

@tool
def rebuild_resume_index() -> str:
    """Rebuild the vector store index from all resume files in data/resumes/."""
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
    vs = build_vector_store()
    count = vs._collection.count()
    return f"Index rebuilt successfully with {count} chunks from resume files."

# ─── Custom Agent Tools ───────────────────────────────────────────────────────

@tool
def extract_requirements(jd: str) -> str:
    """Parse a job description and extract must-have and nice-to-have requirements.

    Args:
        jd: Full text of the job description
    """
    llm = get_llm()
    prompt = f"""Analyze this job description and extract requirements.
Return a JSON object with these keys:
- "role": job title
- "must_have": list of absolutely required skills/experience
- "nice_to_have": list of preferred but optional skills
- "experience_years": minimum years required (integer or null)
- "education": education requirements (string or null)
- "key_responsibilities": top 5 responsibilities as list

Return ONLY valid JSON, no markdown fences.

JOB DESCRIPTION:
{jd}"""

    response = llm.invoke(prompt)
    try:
        text = response.content.strip()
        text = re.sub(r"```json|```", "", text).strip()
        parsed = json.loads(text)
        return json.dumps(parsed, indent=2)
    except Exception:
        return response.content

@tool
def compare_candidates(candidate_ids: str, jd_requirements: str) -> str:
    """Compare multiple candidates head-to-head against job requirements.

    Args:
        candidate_ids: Comma-separated candidate names/IDs (e.g. 'john_doe,jane_smith')
        jd_requirements: JSON string of extracted requirements from extract_requirements tool
    """
    llm = get_llm()
    ids = [c.strip() for c in candidate_ids.split(",")]

    resumes = {}
    for cid in ids:
        for ext in [".txt", ".pdf"]:
            path = RESUME_DIR / f"{cid}{ext}"
            if path.exists():
                if ext == ".pdf":
                    loader = PyPDFLoader(str(path))
                    resumes[cid] = "\n".join(d.page_content for d in loader.load())
                else:
                    resumes[cid] = path.read_text(encoding="utf-8")
                break
        if cid not in resumes:
            matches = list(RESUME_DIR.glob(f"*{cid}*"))
            if matches:
                resumes[cid] = matches[0].read_text(encoding="utf-8")
            else:
                resumes[cid] = f"[Resume not found for {cid}]"

    resume_text = "\n\n===\n\n".join(
        f"CANDIDATE: {cid}\n{text}" for cid, text in resumes.items()
    )

    prompt = f"""Compare these candidates for the role. For each, provide:
1. Overall match score (0-100)
2. Must-have requirements met (list)
3. Must-have requirements missing (list)
4. Nice-to-have requirements met (list)
5. Key strengths
6. Key gaps
7. Hire recommendation: STRONG_YES / YES / MAYBE / NO

JOB REQUIREMENTS:
{jd_requirements}

CANDIDATES:
{resume_text}

Format as a clear comparison table-style text, then a ranking summary."""

    response = llm.invoke(prompt)
    return response.content

@tool
def generate_interview_questions(candidate_id: str, jd_requirements: str) -> str:
    """Generate tailored screening interview questions for a specific candidate.

    Args:
        candidate_id: Candidate name/ID matching their resume filename
        jd_requirements: JSON string of extracted requirements
    """
    llm = get_llm()

    resume = "[Resume not found]"
    for ext in [".txt", ".pdf"]:
        path = RESUME_DIR / f"{candidate_id}{ext}"
        if path.exists():
            if ext == ".pdf":
                loader = PyPDFLoader(str(path))
                resume = "\n".join(d.page_content for d in loader.load())
            else:
                resume = path.read_text(encoding="utf-8")
            break

    if resume == "[Resume not found]":
        matches = list(RESUME_DIR.glob(f"*{candidate_id}*"))
        if matches:
            resume = matches[0].read_text(encoding="utf-8")

    prompt = f"""Generate a tailored interview question set for this candidate.

JOB REQUIREMENTS:
{jd_requirements}

CANDIDATE RESUME:
{resume}

Create 10 interview questions:
- 3 technical depth questions (probe claimed skills)
- 2 gap-probing questions (address missing requirements)
- 2 behavioral questions (STAR format prompts)
- 2 role-specific scenario questions
- 1 culture/motivation question

For each question explain WHY you're asking it given this specific candidate's background."""

    response = llm.invoke(prompt)
    return response.content

@tool
def generate_match_report(candidate_id: str, score: int, jd_requirements: str) -> str:
    """Generate a detailed match report with strengths, gaps, and improvement suggestions.

    Args:
        candidate_id: Candidate name/ID
        score: Match score 0-100
        jd_requirements: JSON string of extracted requirements
    """
    llm = get_llm()

    resume = "[Resume not found]"
    for ext in [".txt", ".pdf"]:
        path = RESUME_DIR / f"{candidate_id}{ext}"
        if path.exists():
            resume = path.read_text(encoding="utf-8")
            break

    prompt = f"""Write a detailed match report for this candidate (score: {score}/100).

JOB REQUIREMENTS:
{jd_requirements}

RESUME:
{resume}

Report must include:
1. **Executive Summary** (2-3 sentences)
2. **Strengths** (bullet list with evidence from resume)
3. **Gaps** (bullet list with severity: critical/minor)
4. **Match Score Breakdown** (per requirement category)
5. **Improvement Suggestions** (for borderline candidates)
6. **Final Recommendation**: HIRE / HOLD / REJECT with reasoning"""

    response = llm.invoke(prompt)
    return response.content


# ─── All tools list (for agent) ───────────────────────────────────────────────
ALL_TOOLS = [
    list_resume_files,
    read_resume,
    list_jd_files,
    read_job_description,
    rag_search_resumes,
    rebuild_resume_index,
    extract_requirements,
    compare_candidates,
    generate_interview_questions,
    generate_match_report,
]