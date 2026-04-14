import streamlit as st
import os
from dotenv import load_dotenv
from src.auth_db import init_db, register_user, verify_user, save_chat, get_chat_history, purge_system_chats
from src.evaluate import run_evaluation_suite
from src.ingestion import load_and_chunk_pdfs
from src.metadata_tagger import enrich_metadata
from src.retrieval_engine import get_vectorstore, create_relationship_aware_rag_chain
from src.api_ingestion import download_act_pdf, search_acts, get_available_acts, fetch_from_any_url

load_dotenv()
init_db()

st.set_page_config(page_title="Relationship-Aware RAG", page_icon="⚖️", layout="wide")

# Hide standard formatting & Implement Clean/White Professional Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
        max-width: 1000px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    div[data-testid="stChatMessage"] {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #E5E7EB;
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
# 1. SIDEBAR (Authentication Profile)
# ==========================================
with st.sidebar:
    if st.session_state['username'] is None:
        st.markdown("### 👤 Access Portal")
        with st.expander("Log in / Sign up", expanded=True):
            tab_login, tab_register = st.tabs(["Login", "Register"])
            
            with tab_login:
                l_user = st.text_input("Username", key="l_u")
                l_pass = st.text_input("Password", type="password", key="l_p")
                if st.button("Authenticate", use_container_width=True, type="primary"):
                    success, role = verify_user(l_user, l_pass)
                    if success:
                        st.session_state['username'] = l_user
                        st.session_state['role'] = role
                        st.rerun()
                    else:
                        st.error("Access Denied.")
                        
            with tab_register:
                r_user = st.text_input("New Entity Name", key="r_u")
                r_pass = st.text_input("Password", type="password", key="r_p")
                r_admin = st.text_input("Admin Key (Optional)", type="password")
                if st.button("Register", use_container_width=True):
                    if r_user and r_pass:
                        is_admin = (r_admin == "ADMIN_123")
                        success, msg = register_user(r_user, r_pass, is_admin)
                        if success: st.success("Verified. Please Sign In.")
                        else: st.error(msg)
        st.write("Browse as Guest implicitly to use the AI offline.")
    else:
        st.markdown(f"### 👤 Welcome, **{st.session_state['username']}**")
        role_color = "green" if st.session_state['role'] == "citizen" else "red"
        st.markdown(f"**Clearance Level:** :{role_color}[`{st.session_state['role'].upper()}`]")
        
        if st.button("Log Off Securely", use_container_width=True):
            st.session_state['username'] = None
            st.session_state['role'] = "guest"
            st.session_state['messages'] = []
            if 'chat_loaded' in st.session_state:
                del st.session_state['chat_loaded']
            st.rerun()
            
    st.divider()

st.markdown("""
<div style="background-color: white; padding: 25px; border-radius: 8px; margin-bottom: 25px; border-left: 5px solid #1E3A8A; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h2 style="color: #1E3A8A; margin: 0; font-family: 'Inter', sans-serif; font-size: 2rem; font-weight: 600;">⚖️ Relationship-Aware Policy Analytics</h2>
    <p style="color: #6B7280; margin: 5px 0 0 0; font-family: 'Inter', sans-serif; font-size: 1.1rem;">RAG-Based GenAI for E-Governance Policy Retrieval</p>
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
    st.markdown("### ⚙️ Engine Administrator Dashboard")
    tab_sync, tab_graph, tab_eval, tab_manage = st.tabs(["🚀 System Sync", "🕸️ Relationship Graph", "📊 Evaluation Lab", "🗄️ File Manager"])
    
    with tab_sync:
        st.write("Manage structural dependencies and synchronize raw PDFs.")
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
                
    with tab_graph:
        if os.path.exists("data/relationship_graph.json"):
            with open("data/relationship_graph.json", "r") as f:
                st.json(f.read(), expanded=False)
        else:
            st.warning("Nodes Offline. Sync Required.")
            
    with tab_eval:
        st.write("Run the 50-question empirical test battery comparing your Relationship-Aware RAG against a Naive Chatbot baseline.")
        if st.button("▶️ Run Full Evaluation (50 Questions)", use_container_width=True, type="primary"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(p, m):
                    progress_bar.progress(p)
                    status_text.text(m)

                results = run_evaluation_suite(update_progress)
                status_text.text("✅ Evaluation Complete!")
                m = results["metrics"]

                st.subheader("📊 Overall Performance")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Naive Chatbot", f"{m['naive_accuracy']:.1f}%", help="Like asking ChatGPT/Gemini directly")
                col2.metric("Aware RAG (Yours)", f"{m['aware_accuracy']:.1f}%", f"+{m['improvement']:.1f}%")
                col3.metric("Hallucination Rate", f"{m['hallucination_rate']:.1f}%")
                col4.metric("Total Questions", m["total_queries"])

                st.subheader("🎯 Amendment-Trap Questions")
                st.write("Questions specifically designed to trip up chatbots that don't track law amendments:")
                col_a, col_b = st.columns(2)
                col_a.metric("Naive Chatbot on Tricky Qs", f"{m['tricky_naive_accuracy']:.1f}%")
                col_b.metric("Aware RAG on Tricky Qs", f"{m['tricky_aware_accuracy']:.1f}%", f"+{m['tricky_aware_accuracy'] - m['tricky_naive_accuracy']:.1f}%")

                st.subheader("📂 Category Breakdown")
                for cat, s in results["category_scores"].items():
                    n_pct = (s["naive"] / s["total"]) * 100
                    a_pct = (s["aware"] / s["total"]) * 100
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                    c1.write(f"**{cat}**")
                    c2.metric("Naive", f"{n_pct:.0f}%")
                    c3.metric("Aware", f"{a_pct:.0f}%", f"+{a_pct - n_pct:.0f}%")
                    c4.write(f"{s['total']} Qs")

                st.subheader("🔍 Question-by-Question Breakdown")
                for res in results["breakdown"]:
                    icon_n = "✅" if res["naive_pass"] else "❌"
                    icon_a = "✅" if res["aware_pass"] else "❌"
                    trap = "🎯 TRICKY" if res["tricky"] else ""
                    st.markdown(f"**[{res['category']}] {trap}** {res['query']}\n- Chatbot: {icon_n} | RAG: {icon_a}")
                    st.divider()
            except Exception as e:
                st.error(f"Evaluation Failed: {e}")

        
    with tab_manage:
        st.write("Upload and examine loaded policy structures.")
        
        # --- India Code Library ---
        st.subheader("🌐 India Code Library")
        st.write("Search and download official Acts directly from India Code (indiacode.nic.in).")
        search_q = st.text_input("Search Acts", placeholder="e.g., RTI, Environment, CGST...")
        acts_dict = search_acts(search_q) if search_q else get_available_acts()
        
        if acts_dict:
            for act_name, act_url in acts_dict.items():
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    st.markdown(f"📜 **{act_name}**")
                with col2:
                    if st.button("Base", key=f"base_{act_name}", help="Download as a base/original act"):
                        with st.spinner(f"Downloading {act_name}..."):
                            ok, msg = download_act_pdf(act_name, act_url, is_amendment=False)
                        if ok:
                            st.success(f"Saved! Run System Sync to index it.")
                        else:
                            st.error(f"Failed: {msg}")
                with col3:
                    if st.button("Amend", key=f"amend_{act_name}", help="Download as an amendment act"):
                        with st.spinner(f"Downloading {act_name}..."):
                            ok, msg = download_act_pdf(act_name, act_url, is_amendment=True)
                        if ok:
                            st.success(f"Saved! Run System Sync to index it.")
                        else:
                            st.error(f"Failed: {msg}")
        else:
            st.info("No acts found matching your search.")

        st.divider()
        st.subheader("🔗 Live Web Fetch (Paste any India Code URL)")
        st.write("Paste any URL from `indiacode.nic.in` — the system will scrape and download the PDF automatically.")
        col_url, col_name = st.columns([3, 2])
        with col_url:
            paste_url = st.text_input("India Code / Direct PDF URL", placeholder="https://www.indiacode.nic.in/handle/123456789/...")
        with col_name:
            paste_name = st.text_input("Short Name for File", placeholder="e.g., motor_vehicles_act")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("⬇️ Fetch as Base", use_container_width=True):
                if paste_url and paste_name:
                    with st.spinner("Fetching PDF from India Code..."):
                        ok, msg = fetch_from_any_url(paste_url, paste_name, is_amendment=False)
                    st.success(f"Saved as `base_{paste_name}.pdf`! Run Sync to index.") if ok else st.error(msg)
                else:
                    st.warning("Please enter both a URL and a short name.")
        with col_b2:
            if st.button("⬇️ Fetch as Amendment", use_container_width=True):
                if paste_url and paste_name:
                    with st.spinner("Fetching PDF from India Code..."):
                        ok, msg = fetch_from_any_url(paste_url, paste_name, is_amendment=True)
                    st.success(f"Saved as `amendment_{paste_name}.pdf`! Run Sync to index.") if ok else st.error(msg)
                else:
                    st.warning("Please enter both a URL and a short name.")

        st.divider()
        
        # --- File Upload Section ---
        st.subheader("Bulk File Ingestion")
        uploaded_files = st.file_uploader("Upload new Policy PDFs", accept_multiple_files=True, type=['pdf'])
        if uploaded_files:
            if st.button(f"Acquire {len(uploaded_files)} Document(s)", type="primary"):
                os.makedirs("data/raw", exist_ok=True)
                for f in uploaded_files:
                    with open(os.path.join("data/raw", f.name), "wb") as f_out:
                        f_out.write(f.getbuffer())
                st.success(f"Successfully saved {len(uploaded_files)} files to raw storage! Please run 'System Sync' to index them.")
                
        st.divider()
        st.subheader("Raw Storage Index")
        if os.path.exists("data/raw"):
            files = os.listdir("data/raw")
            pdf_files = [f for f in files if f.endswith(".pdf")]
            st.metric("Total Raw Documents", len(pdf_files))
            with st.expander("View Document Index"):
                for pdf in pdf_files:
                    st.markdown(f"- 📄 `{pdf}`")
        else:
            st.warning("No raw data directory found.")
            
        st.divider()
        st.subheader("Rename Document")
        st.write("Use explicitly formatted names (e.g., `base_policy.pdf`, `amendment_policy_2024.pdf`) for guaranteed relationship mapping.")
        if os.path.exists("data/raw") and pdf_files:
            col_a, col_b = st.columns(2)
            with col_a:
                old_name = st.selectbox("Select File", pdf_files)
            with col_b:
                new_name = st.text_input("New Name (must end in .pdf)", value=old_name)
                
            if st.button("Apply New Name", use_container_width=True):
                if new_name and new_name.endswith(".pdf") and new_name != old_name:
                    try:
                        os.rename(os.path.join("data/raw", old_name), os.path.join("data/raw", new_name))
                        st.success(f"Renamed `{old_name}` to `{new_name}`!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to rename: {e}")
                elif not new_name.endswith(".pdf"):
                    st.error("Filename must end with .pdf")
                    
        st.divider()
        st.write("⚠️ **Danger Zone: Privacy Options**")
        st.write("Wipe all citizen interaction histories from the SQLite database to comply with Data Deletion requests.")
        if st.button("🗑️ Purge System Chats", type="primary", use_container_width=True):
            try:
                purge_system_chats()
                st.success("All interaction logs securely purged.")
            except Exception as e:
                st.error(f"Purge Failure: {e}")
