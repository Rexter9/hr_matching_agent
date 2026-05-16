"""
test_scenarios.py — 5+ Conversation Flow Tests for the HR Matching Agent
Run with: python test_scenarios.py

Each scenario tests a different aspect of the agent:
1. Full matching workflow (end-to-end)
2. Natural language candidate search
3. Iterative refinement / re-ranking
4. Head-to-head comparison
5. Explainability ("Why did X rank higher than Y?")
6. Multi-round screening
"""
import uuid, time, sys
from dotenv import load_dotenv

load_dotenv()

from matching_agent import run_agent, continue_conversation

# ─── Helpers ──────────────────────────────────────────────────────────────────

def new_thread():
    return f"test_{uuid.uuid4().hex[:6]}"

def chat(thread_id, prompt, label=""):
    print(f"\n{'='*60}")
    print(f"USER: {prompt}")
    print(f"{'='*60}")
    try:
        response = continue_conversation(prompt, thread_id=thread_id)
        print(f"AGENT: {response[:1200]}{'...' if len(response)>1200 else ''}")
    except Exception as e:
        print(f"ERROR: {e}")
    print()
    return response if 'response' in dir() else ""

def scenario_header(n, title):
    print(f"\n\n{'#'*70}")
    print(f"  SCENARIO {n}: {title}")
    print(f"{'#'*70}")

# ─── Scenario 1: Full Matching Workflow ───────────────────────────────────────

def scenario_1_full_workflow():
    scenario_header(1, "Full End-to-End Matching Workflow")
    tid = new_thread()
    
    # Initialize
    run_agent(
        "You are starting a new recruiting session. List available JD files and resumes.",
        thread_id=tid, stage="agent"
    )
    
    chat(tid, "Load the first available job description and extract all requirements from it.")
    time.sleep(1)
    chat(tid, "Now search all resumes for candidates that match these requirements.")
    time.sleep(1)
    chat(tid, "Rank the top 5 candidates with match scores and brief reasoning.")
    time.sleep(1)
    chat(tid, "Generate a full hiring report for the #1 ranked candidate including interview questions.")

# ─── Scenario 2: Natural Language Search ──────────────────────────────────────

def scenario_2_natural_language_search():
    scenario_header(2, "Natural Language Candidate Search")
    tid = new_thread()
    
    run_agent("Hello, I need help finding candidates.", thread_id=tid, stage="agent")
    
    chat(tid, "Find me candidates with Python and at least 3 years of experience.")
    chat(tid, "Now find candidates who have experience with React AND TypeScript.")
    chat(tid, "Which candidates have both backend and frontend skills?")
    chat(tid, "Show me candidates who have worked at startups or have leadership experience.")

# ─── Scenario 3: Iterative Refinement / Re-ranking ────────────────────────────

def scenario_3_iterative_refinement():
    scenario_header(3, "Iterative Refinement and Re-ranking")
    tid = new_thread()
    
    run_agent("Start a matching workflow.", thread_id=tid, stage="agent")
    
    chat(tid, "Search resumes and rank candidates for a Python backend developer role.")
    time.sleep(1)
    chat(tid, "Now re-rank them but this time prioritize AWS experience over Python seniority.")
    time.sleep(1)
    chat(tid, "Actually, the client says SQL knowledge is now a hard requirement. Re-rank again.")
    time.sleep(1)
    chat(tid, "What changed in the rankings between the first and last ranking? Explain the differences.")

# ─── Scenario 4: Head-to-Head Comparison ──────────────────────────────────────

def scenario_4_comparison():
    scenario_header(4, "Head-to-Head Candidate Comparison")
    tid = new_thread()
    
    run_agent("I want to compare specific candidates.", thread_id=tid, stage="agent")
    
    # List resumes first to get real names
    chat(tid, "List all available resume files so I know the candidate names.")
    time.sleep(1)
    chat(tid, "Compare the top 3 candidates from your resume list side by side. "
              "Use the compare_candidates tool with their names.")
    time.sleep(1)
    chat(tid, "Which candidate is stronger technically? Which is stronger in leadership?")
    chat(tid, "If I had to pick one for a senior IC role vs one for a team lead role, who would you recommend for each?")

# ─── Scenario 5: Explainability ───────────────────────────────────────────────

def scenario_5_explainability():
    scenario_header(5, "Explainability — Why Did X Rank Higher?")
    tid = new_thread()
    
    run_agent(
        "Rank the top 3 candidates for a Full Stack Developer role with 5+ years experience.",
        thread_id=tid, stage="agent"
    )
    time.sleep(1)
    
    chat(tid, "Why did the top ranked candidate score higher than the second ranked candidate? "
              "Give me specific evidence from their resumes.")
    time.sleep(1)
    chat(tid, "What does the third-ranked candidate need to improve to become the top choice?")
    chat(tid, "Is the current #1 definitely the best choice, or are there any hidden risks I should know about?")
    chat(tid, "Generate tailored interview questions for the #1 candidate that specifically probe their weakest areas.")

# ─── Scenario 6: Multi-Round Screening ────────────────────────────────────────

def scenario_6_multi_round():
    scenario_header(6, "Multi-Round Screening (3 Rounds)")
    tid = new_thread()
    
    run_agent("Start a multi-round screening process.", thread_id=tid, stage="agent")
    
    chat(tid, "ROUND 1 - Initial Screen: Search all resumes and identify the top candidates "
              "based on keyword and skills match. Give me a preliminary shortlist of up to 5.")
    time.sleep(1)
    
    chat(tid, "ROUND 2 - Deep Analysis: Take the top 3 from round 1 and do a detailed analysis. "
              "Generate full match reports for each using the generate_match_report tool.")
    time.sleep(1)
    
    chat(tid, "ROUND 3 - Final Decision: Based on the deep analysis, give me a final hire/no-hire "
              "recommendation for each finalist with full justification. "
              "Who should I bring in for an in-person interview?")

# ─── Main ─────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "1": ("Full Workflow", scenario_1_full_workflow),
    "2": ("Natural Language Search", scenario_2_natural_language_search),
    "3": ("Iterative Refinement", scenario_3_iterative_refinement),
    "4": ("Head-to-Head Comparison", scenario_4_comparison),
    "5": ("Explainability", scenario_5_explainability),
    "6": ("Multi-Round Screening", scenario_6_multi_round),
}

if __name__ == "__main__":
    print("\n🎯 HR Matching Agent — Test Scenarios")
    print("="*50)
    
    if len(sys.argv) > 1:
        # Run specific scenario
        n = sys.argv[1]
        if n in SCENARIOS:
            name, fn = SCENARIOS[n]
            print(f"Running Scenario {n}: {name}")
            fn()
        else:
            print(f"Unknown scenario: {n}. Options: {list(SCENARIOS.keys())}")
    else:
        # Run all
        print("Running ALL scenarios...")
        print("Tip: Run a single scenario with: python test_scenarios.py 1\n")
        for n, (name, fn) in SCENARIOS.items():
            fn()
            time.sleep(2)
    
    print("\n\n✅ All test scenarios complete!")
