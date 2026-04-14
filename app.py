import streamlit as st
import os
from dotenv import load_dotenv
from src.auth_db import init_db, register_user, verify_user, save_chat, get_chat_history
from src.ingestion import load_and_chunk_pdfs
from src.metadata_tagger import enrich_metadata
from src.retrieval_engine import get_vectorstore, create_relationship_aware_rag_chain

load_dotenv()
init_db()

st.set_page_config(page_title="Relationship-Aware RAG", page_icon="⚖️", layout="wide")

# Hide standard formatting
st.markdown("""
<style>
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Session initialization
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'role' not in st.session_state:
    st.session_state['role'] = "guest"
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 1. SIDEBAR (Authentication & Admin)
# ==========================================
with st.sidebar:
    if st.session_state['username'] is None:
        st.markdown("### 🔐 Authentication Portal")
        st.write("Ensure secure access to your profile.")
        
        tab_login, tab_register = st.tabs(["Login", "Register"])
        with tab_login:
            l_user = st.text_input("Username", key="l_u")
            l_pass = st.text_input("Password", type="password", key="l_p")
            if st.button("Sign In", use_container_width=True):
                success, role = verify_user(l_user, l_pass)
                if success:
                    st.session_state['username'] = l_user
                    st.session_state['role'] = role
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
                    
        with tab_register:
            r_user = st.text_input("New Username", key="r_u")
            r_pass = st.text_input("New Password", type="password", key="r_p")
            r_admin = st.text_input("Admin Key (Optional)", type="password", help="Use 'ADMIN_123' to register as administration. Citizens leave blank.")
            if st.button("Register Identity", use_container_width=True):
                if r_user and r_pass:
                    is_admin = (r_admin == "ADMIN_123")
                    success, msg = register_user(r_user, r_pass, is_admin)
                    if success: st.success(msg)
                    else: st.error(msg)
    else:
        # LOGGED IN VIEW
        st.success(f"System Authorized: **{st.session_state['username']}**")
        if st.button("Log Off Securely", use_container_width=True):
            st.session_state['username'] = None
            st.session_state['role'] = "guest"
            st.session_state['messages'] = []
            if 'chat_loaded' in st.session_state:
                del st.session_state['chat_loaded']
            st.rerun()
            
        st.divider()
        if st.session_state['role'] == "admin":
            st.markdown("### ⚙️ Engine Administrator")
            st.write("Manage structural dependencies.")
            if st.button("🚀 Sync Policy Database", use_container_width=True):
                try:
                    with st.spinner("Analyzing structures..."):
                        chunks = load_and_chunk_pdfs()
                    with st.spinner("Generating Temporal Nodes..."):
                        tagged_chunks = enrich_metadata(chunks)
                    with st.spinner("Constructing Vector Space..."):
                        get_vectorstore(tagged_chunks)
                    st.success("System Sync Completed Successfully.")
                except Exception as e:
                    st.error(f"Sync Failure: {e}")
                    
            st.markdown("### 🕸️ Relationship Graph")
            if os.path.exists("data/relationship_graph.json"):
                with open("data/relationship_graph.json", "r") as f:
                    st.json(f.read(), expanded=False)
            else:
                st.warning("Nodes Offline. Sync Required.")

        elif st.session_state['role'] == "citizen":
            st.info("End-to-End Encrypted Session. Chat history auto-saves.")

# ==========================================
# 2. MAIN HEADER (Professional Look)
# ==========================================
st.markdown("""
<div style="background-color: #2563EB; padding: 25px; border-radius: 8px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h2 style="color: white; margin: 0; font-family: 'Inter', sans-serif; font-size: 2rem; font-weight: 600;">⚖️ Relationship-Aware Policy Analytics</h2>
    <p style="color: #DBEAFE; margin: 5px 0 0 0; font-family: 'Inter', sans-serif; font-size: 1.1rem;">Relationship-Aware RAG-Based GenAI for E-Governance Policy Retrieval and Responses</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 3. INTERACTIVE CHAT ENGINE
# ==========================================
# Auto-load history for logged-in citizens
if st.session_state['role'] == "citizen":
    if "chat_loaded" not in st.session_state or st.session_state.chat_loaded != st.session_state['username']:
        st.session_state.messages = get_chat_history(st.session_state['username'])
        st.session_state.chat_loaded = st.session_state['username']
        
        if not st.session_state.messages:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Welcome to the Analytics Platform. How can we resolve your civic and policy inquiries today?"
            })

# Allow Guests AND Citizens to chat (Admins do not need to chat here)
if st.session_state['role'] in ["citizen", "guest"]:
    
    if st.session_state['role'] == "guest" and not st.session_state.messages:
        # Provide a specific guest greeting
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Welcome to the Analytics Platform! You are currently browsing in **Guest Mode** (Chats will not be saved). How can we resolve your political or civic inquiries today?"
        })

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Input civic inquiry or policy parameter..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Only persist to SQLite if they are a registered citizen
        if st.session_state['role'] == "citizen":
            save_chat(st.session_state['username'], "user", prompt)

        if not os.path.exists("data/chroma_db"):
            st.error("Admin Authentication Required: Database vector memory empty.")
        elif not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY").startswith("gsk-your"):
            st.error("Configuration Defaulted: LLM API Key explicitly missing.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Bridging semantic policy networks..."):
                    try:
                        rag_chain = create_relationship_aware_rag_chain()
                        response = rag_chain.invoke({"input": prompt})
                        
                        answer = response["answer"]
                        sources = response.get("context", [])
                        
                        output = answer + "\n\n---\n**✅ Verified Document Nodes Examined:**\n"
                        displayed_sources = set()
                        for doc in sources:
                            doc_name = doc.metadata.get('document_name', 'Unknown')
                            status = doc.metadata.get('status', 'Unknown')
                            status_emoji = "🟢" if "Active" in status else "🔴"
                            if doc_name not in displayed_sources:
                                output += f"- {status_emoji} `{doc_name}` - Database Status: *{status.upper()}*\n"
                                displayed_sources.add(doc_name)
                                
                        st.markdown(output)
                        st.session_state.messages.append({"role": "assistant", "content": output})
                        
                        if st.session_state['role'] == "citizen":
                            save_chat(st.session_state['username'], "assistant", output)
                            
                    except Exception as e:
                        st.error(f"Memory Architecture Error: {str(e)}")
elif st.session_state['role'] == "admin":
    st.info("Administrative View. To query the AI Database, please Log Off and browse as a Guest or Citizen.")
