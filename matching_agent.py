"""
matching_agent.py — LangGraph-based HR Matching Agent
Implements the full state machine:
START → Parse JD → Extract Requirements → Search Resumes →
Rank Candidates → Generate Report → Human Feedback Loop → END
"""
import json
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.messages import ToolMessage
import operator
from dotenv import load_dotenv

load_dotenv()

from tools import ALL_TOOLS

# ─── Agent State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Conversation history (all messages)
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Job description understanding
    jd_text: Optional[str]
    jd_filename: Optional[str]
    requirements: Optional[dict]   # parsed must-have / nice-to-have
    
    # Candidate tracking
    candidate_shortlist: List[str]  # candidate IDs
    candidate_scores: dict          # {candidate_id: score}
    candidate_reasoning: dict       # {candidate_id: reasoning text}
    
    # Workflow state
    current_stage: str              # which stage we're in
    screening_round: int            # 1=initial, 2=deep, 3=final
    final_report: Optional[str]
    
    # Human feedback
    awaiting_feedback: bool
    user_refinements: List[str]     # history of refinements

# ─── LLM Setup ────────────────────────────────────────────────────────────────

def get_agent_llm():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return llm.bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = """You are an expert HR recruiting agent. Your job is to match candidates to job requirements.

You have access to these tools:
- list_resume_files, read_resume: file system tools for resumes
- list_jd_files, read_job_description: file system tools for JDs
- rag_search_resumes: semantic search across all resumes
- rebuild_resume_index: index resumes for RAG search
- extract_requirements: parse JD into must-have/nice-to-have
- compare_candidates: head-to-head comparison
- generate_interview_questions: tailored screening questions
- generate_match_report: detailed match analysis

WORKFLOW:
1. When given a JD file or text, first extract requirements
2. Search resumes semantically for top matches
3. Rank candidates with reasoning
4. Generate detailed reports for top candidates
5. Accept user refinements and re-rank accordingly

TONE: Be precise, evidence-based, and explain your reasoning.
When ranking, always cite specific skills/experience from resumes.
For borderline candidates, provide constructive improvement suggestions."""

# ─── Graph Nodes ──────────────────────────────────────────────────────────────

def parse_jd_node(state: AgentState) -> AgentState:
    """Parse the job description from state messages."""
    llm = get_agent_llm()
    
    # Build context message
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    # Add instruction if this is the initial JD parsing
    if state.get("current_stage") == "parse_jd":
        instr = HumanMessage(content=(
            "First, list available JD files and resume files, "
            "then read the job description. "
            "Extract all requirements using extract_requirements tool."
        ))
        messages.append(instr)
    
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "current_stage": "extract_requirements"
    }

def extract_requirements_node(state: AgentState) -> AgentState:
    """Extract structured requirements from the parsed JD."""
    llm = get_agent_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "current_stage": "search_resumes"
    }

def search_resumes_node(state: AgentState) -> AgentState:
    """Search resumes using RAG and build initial shortlist."""
    llm = get_agent_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    # Add search instruction
    if state.get("current_stage") == "search_resumes":
        # Build query from requirements if available
        req = state.get("requirements", {})
        skills = req.get("must_have", []) if req else []
        skill_str = ", ".join(skills[:5]) if skills else "relevant skills"
        
        instr = HumanMessage(content=(
            f"Now search resumes for candidates with: {skill_str}. "
            "Use rag_search_resumes to find the best matches. "
            "Also list all available resume files."
        ))
        messages.append(instr)
    
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "current_stage": "rank_candidates"
    }

def rank_candidates_node(state: AgentState) -> AgentState:
    """Rank candidates and build shortlist with reasoning."""
    llm = get_agent_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    round_num = state.get("screening_round", 1)
    
    if round_num == 1:
        instr = HumanMessage(content=(
            "Based on the search results, compare the top candidates using "
            "compare_candidates tool. Rank them 1-10 with scores. "
            "Explain your reasoning for each ranking."
        ))
    elif round_num == 2:
        instr = HumanMessage(content=(
            "Deep analysis round: For the top 3 candidates, "
            "generate detailed match reports using generate_match_report. "
            "Compare them head-to-head."
        ))
    else:
        instr = HumanMessage(content=(
            "Final round: Generate hire/no-hire recommendations "
            "with full justification for each finalist."
        ))
    
    messages.append(instr)
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "current_stage": "generate_report",
        "screening_round": round_num
    }

def generate_report_node(state: AgentState) -> AgentState:
    """Generate final report for top candidates."""
    llm = get_agent_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    instr = HumanMessage(content=(
        "Generate a comprehensive hiring report. Include: "
        "1) Ranked shortlist with scores "
        "2) Top candidate analysis "
        "3) Interview questions for the #1 candidate using generate_interview_questions "
        "4) Summary recommendation for the hiring manager. "
        "Be specific and cite evidence from resumes."
    ))
    messages.append(instr)
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "current_stage": "human_feedback",
        "awaiting_feedback": True
    }

def agent_node(state: AgentState) -> AgentState:
    """General conversational agent node — handles user queries and refinements."""
    llm = get_agent_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "awaiting_feedback": False
    }

# ─── Routing Logic ────────────────────────────────────────────────────────────

def should_use_tools(state: AgentState) -> str:
    """Route to tool execution if the last message has tool calls."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end_node"

def route_by_stage(state: AgentState) -> str:
    """Route to the appropriate next stage."""
    stage = state.get("current_stage", "agent")
    stage_map = {
        "parse_jd":             "parse_jd",
        "extract_requirements": "extract_requirements",
        "search_resumes":       "search_resumes",
        "rank_candidates":      "rank_candidates",
        "generate_report":      "generate_report",
        "human_feedback":       "agent",
    }
    return stage_map.get(stage, "agent")

# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_agent_graph():
    graph = StateGraph(AgentState)
    
    # Tool execution node
    tool_node = ToolNode(ALL_TOOLS)
    
    # Add all nodes
    graph.add_node("agent",                agent_node)
    graph.add_node("parse_jd",             parse_jd_node)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes",       search_resumes_node)
    graph.add_node("rank_candidates",      rank_candidates_node)
    graph.add_node("generate_report",      generate_report_node)
    graph.add_node("tools",                tool_node)
    graph.add_node("end_node",             lambda s: s)
    
    # Entry point
    graph.set_entry_point("agent")
    
    # Agent → tools or end
    graph.add_conditional_edges(
        "agent",
        should_use_tools,
        {"tools": "tools", "end_node": END}
    )
    
    # Tools loop back to agent
    graph.add_edge("tools", "agent")
    
    # Workflow nodes → tools or end
    for node in ["parse_jd", "extract_requirements", "search_resumes",
                 "rank_candidates", "generate_report"]:
        graph.add_conditional_edges(
            node,
            should_use_tools,
            {"tools": "tools", "end_node": END}
        )
    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# ─── Public API ───────────────────────────────────────────────────────────────

# Singleton graph
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph

def run_agent(user_message: str, thread_id: str = "default",
              stage: str = "agent") -> str:
    """
    Run the agent with a user message.
    
    Args:
        user_message: Natural language input from the user
        thread_id: Conversation thread ID (for multi-turn memory)
        stage: Which workflow stage to start from ('agent' for free-form chat)
    
    Returns:
        Agent's response text
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    state = {
        "messages": [HumanMessage(content=user_message)],
        "jd_text": None,
        "jd_filename": None,
        "requirements": None,
        "candidate_shortlist": [],
        "candidate_scores": {},
        "candidate_reasoning": {},
        "current_stage": stage,
        "screening_round": 1,
        "final_report": None,
        "awaiting_feedback": False,
        "user_refinements": [],
    }
    
    result = graph.invoke(state, config=config)
    
    # Extract the last AI message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not hasattr(msg, "tool_calls"):
            return msg.content
        if isinstance(msg, AIMessage):
            if not msg.tool_calls:
                return msg.content
    
    # Fallback: get last non-tool message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content
    
    return "Agent completed without a text response."

def continue_conversation(user_message: str, thread_id: str = "default") -> str:
    """
    Continue an existing conversation with full history (via checkpointer).
    
    Args:
        user_message: New user message
        thread_id: Must match the thread_id used in run_agent
    
    Returns:
        Agent's response text
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    # Just add the new message; graph checkpointer handles full history
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config
    )
    
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return msg.content
    
    return "Agent completed."


if __name__ == "__main__":
    # Quick smoke test
    print("Starting HR Matching Agent...")
    print("Type 'quit' to exit.\n")
    
    thread_id = "cli_session"
    
    # Initial startup
    response = run_agent(
        "Hello! List available job descriptions and resumes, "
        "then briefly explain what you can help with.",
        thread_id=thread_id
    )
    print(f"Agent: {response}\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not user_input:
            continue
        
        response = continue_conversation(user_input, thread_id=thread_id)
        print(f"\nAgent: {response}\n")
