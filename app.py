"""
app.py — Streamlit Chat Interface for HR Matching Agent
Run with: streamlit run app.py
"""
import streamlit as st
import uuid
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Matching Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Inline CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.agent-badge {
    background: #1a1a2e; color: #e94560; padding: 2px 8px;
    border-radius: 4px; font-size: 11px; font-weight: 600;
    letter-spacing: 1px; margin-bottom: 4px; display: inline-block;
}
.stage-chip {
    background: #16213e; color: #0f3460; border: 1px solid #0f3460;
    padding: 2px 10px; border-radius: 12px; font-size: 11px;
    display: inline-block; margin: 2px;
}
.stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Imports (after load_dotenv) ──────────────────────────────────────────────
try:
    from matching_agent import run_agent, continue_conversation, get_graph
    from tools import rebuild_resume_index, JD_DIR, RESUME_DIR, ALL_TOOLS
    AGENT_LOADED = True
except ImportError as e:
    AGENT_LOADED = False
    IMPORT_ERROR = str(e)

# ─── Session State ────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "workflow_stage" not in st.session_state:
    st.session_state.workflow_stage = "Chat"

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 HR Matching Agent")
    st.caption(f"Session: `{st.session_state.thread_id}`")
    
    st.divider()
    st.subheader("📁 Data Files")
    
    # Show JD files
    jd_files = list(JD_DIR.glob("*")) if AGENT_LOADED else []
    st.write(f"**Job Descriptions** ({len(jd_files)})")
    for f in jd_files:
        st.caption(f"📄 {f.name}")
    if not jd_files:
        st.warning("Add .txt files to `data/job_descriptions/`")
    
    # Show resume files
    resume_files = list(RESUME_DIR.glob("*")) if AGENT_LOADED else []
    st.write(f"**Resumes** ({len(resume_files)})")
    for f in resume_files:
        st.caption(f"👤 {f.name}")
    if not resume_files:
        st.warning("Add .txt files to `data/resumes/`")
    
    st.divider()
    st.subheader("⚙️ Actions")
    
    if st.button("🔄 Rebuild Vector Index", use_container_width=True):
        with st.spinner("Indexing resumes..."):
            if AGENT_LOADED:
                result = rebuild_resume_index.invoke({})
                st.success(result)
            else:
                st.error("Agent not loaded")
    
    if st.button("🧹 New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.session_state.initialized = False
        st.rerun()
    
    st.divider()
    st.subheader("💬 Quick Commands")
    
    quick_prompts = [
        "List all available resumes",
        "Find candidates with Python and 3+ years experience",
        "Compare the top 3 matches side by side",
        "Start full matching workflow for the first JD",
        "Generate interview questions for the top candidate",
        "Why did the top candidate rank highest?",
        "Re-rank prioritizing React over Python",
        "Show hire/no-hire recommendations",
    ]
    
    for prompt in quick_prompts:
        if st.button(f"→ {prompt[:45]}", use_container_width=True, key=f"qp_{prompt[:20]}"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.pending_prompt = prompt
            st.rerun()
    
    st.divider()
    st.subheader("🔀 Workflow Stage")
    stage = st.selectbox(
        "Start next message as:",
        ["Chat", "Parse JD", "Search Resumes", "Rank Candidates", "Generate Report"],
        index=0,
        label_visibility="collapsed"
    )
    STAGE_MAP = {
        "Chat": "agent",
        "Parse JD": "parse_jd",
        "Search Resumes": "search_resumes",
        "Rank Candidates": "rank_candidates",
        "Generate Report": "generate_report",
    }
    st.session_state.workflow_stage = STAGE_MAP[stage]

# ─── Main Chat Area ───────────────────────────────────────────────────────────
st.header("HR Candidate Matching Agent", divider="gray")

if not AGENT_LOADED:
    st.error(f"❌ Could not load agent: {IMPORT_ERROR}")
    st.info("Make sure you've run `pip install -r requirements.txt` and set OPENAI_API_KEY in `.env`")
    st.stop()

# Agent capabilities info
if not st.session_state.messages:
    with st.expander("ℹ️ What can this agent do?", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🔍 Search & Match**\n- Semantic resume search\n- Multi-round screening\n- RAG-powered retrieval")
        with col2:
            st.markdown("**📊 Analysis**\n- Head-to-head comparison\n- Skills gap analysis\n- Match scoring (0-100)")
        with col3:
            st.markdown("**📝 Reports**\n- Detailed match reports\n- Interview questions\n- Hire/no-hire rec.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🎯" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# Handle pending prompt from quick buttons
if "pending_prompt" in st.session_state:
    pending = st.session_state.pending_prompt
    del st.session_state.pending_prompt
    
    with st.chat_message("assistant", avatar="🎯"):
        with st.spinner("Agent thinking..."):
            try:
                if not st.session_state.initialized:
                    response = run_agent(
                        pending,
                        thread_id=st.session_state.thread_id,
                        stage=st.session_state.workflow_stage
                    )
                    st.session_state.initialized = True
                else:
                    response = continue_conversation(
                        pending,
                        thread_id=st.session_state.thread_id
                    )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

# Chat input
if prompt := st.chat_input("Ask the agent... (e.g. 'Find me candidates with React and 3+ years experience')"):
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get agent response
    with st.chat_message("assistant", avatar="🎯"):
        with st.spinner("Agent thinking..."):
            try:
                if not st.session_state.initialized:
                    response = run_agent(
                        prompt,
                        thread_id=st.session_state.thread_id,
                        stage=st.session_state.workflow_stage
                    )
                    st.session_state.initialized = True
                else:
                    response = continue_conversation(
                        prompt,
                        thread_id=st.session_state.thread_id
                    )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                err = f"❌ Agent error: {str(e)}\n\nCheck that OPENAI_API_KEY is set in your .env file."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
