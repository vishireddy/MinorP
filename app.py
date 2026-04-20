import streamlit as st
import os
from dotenv import load_dotenv
from src.auth_db import init_db, register_user, verify_user, save_chat, get_chat_history, purge_system_chats
from src.evaluate import run_evaluation_suite, run_ragas_evaluation
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
        st.write("Run the 50-question empirical test battery comparing three modes: **Naive LLM** (ChatGPT-style) vs **Naive RAG** (basic retrieval) vs **Aware RAG** (yours).")

        # Sync warning
        raw_count = len([f for f in os.listdir("data/raw") if f.endswith(".pdf")]) if os.path.exists("data/raw") else 0
        if raw_count > 0 and not os.path.exists("data/chroma_db"):
            st.warning("⚠️ Vector database not found. Run System Sync before evaluating.")
        elif raw_count > 0:
            st.info(f"📦 {raw_count} PDFs in raw storage. If you recently added files, run System Sync first to index them.")

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

                st.markdown("---")
                st.subheader("📊 Overall Performance  *(Pass = Judge Score ≥ 6/10)*")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "🤖 Gemma2-9b (Google)",
                    f"{m['naive_llm_accuracy']:.1f}%",
                    help="Google's Gemma2-9b via Groq — no document retrieval, pure training knowledge"
                )
                col2.metric(
                    "📄 Mixtral-8x7b (Basic RAG)",
                    f"{m['naive_rag_accuracy']:.1f}%",
                    delta=f"{m['naive_rag_accuracy'] - m['naive_llm_accuracy']:+.1f}% vs Gemma2",
                    help="Mistral Mixtral retrieves documents but ignores amendment relationships"
                )
                col3.metric(
                    "⚖️ LLaMA3 Aware RAG (Ours)",
                    f"{m['aware_accuracy']:.1f}%",
                    delta=f"{m['rag_improvement_over_llm']:+.1f}% vs Gemma2",
                    help="Meta LLaMA3.1 with relationship-aware retrieval and amendment injection"
                )
                col4.metric("Total Questions", m["total_queries"])

                st.markdown("#### 🏅 Average Judge Score  *(out of 10)*")
                c1, c2, c3 = st.columns(3)
                c1.metric("Gemma2-9b", f"{m['naive_llm_avg_score']:.2f}/10")
                c2.metric("Mixtral Basic RAG", f"{m['naive_rag_avg_score']:.2f}/10",
                          delta=f"{m['naive_rag_avg_score'] - m['naive_llm_avg_score']:+.2f}")
                c3.metric("LLaMA3 Aware RAG", f"{m['aware_avg_score']:.2f}/10",
                          delta=f"{m['aware_avg_score'] - m['naive_llm_avg_score']:+.2f}")

                st.markdown("---")
                st.subheader("🎯 Amendment-Trap Questions")
                st.caption("Questions specifically designed to expose chatbots that don't track law amendments")
                c1, c2, c3 = st.columns(3)
                c1.metric("Gemma2-9b", f"{m['tricky_naive_llm_accuracy']:.1f}%")
                c2.metric("Mixtral RAG", f"{m['tricky_naive_rag_accuracy']:.1f}%",
                          delta=f"{m['tricky_naive_rag_accuracy'] - m['tricky_naive_llm_accuracy']:+.1f}%")
                c3.metric("Aware RAG", f"{m['tricky_aware_accuracy']:.1f}%",
                          delta=f"{m['tricky_aware_accuracy'] - m['tricky_naive_llm_accuracy']:+.1f}%")

                st.markdown("---")
                st.subheader("📂 Category Breakdown")
                for cat, s in results["category_scores"].items():
                    t = s["total"]
                    nl = s["naive_llm_pass"] / t * 100
                    nr = s["naive_rag_pass"] / t * 100
                    aw = s["aware_pass"] / t * 100
                    c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
                    c1.write(f"**{cat}**")
                    c2.metric("Gemma2", f"{nl:.0f}%")
                    c3.metric("Mixtral", f"{nr:.0f}%", delta=f"{nr-nl:+.0f}%")
                    c4.metric("Aware", f"{aw:.0f}%", delta=f"{aw-nl:+.0f}%")
                    c5.caption(f"{t} Qs")

                st.markdown("---")
                st.subheader("🔍 Question-by-Question Breakdown  *(with Judge Reasoning)*")

                def score_badge(score):
                    if score >= 8: return f"🟢 {score}/10"
                    if score >= 6: return f"🟡 {score}/10"
                    return f"🔴 {score}/10"

                for i, res in enumerate(results["breakdown"]):
                    trap_tag = " 🎯 **AMENDMENT TRAP**" if res["tricky"] else ""
                    label = f"**Q{i+1}. [{res['category']}]**{trap_tag} — {res['query']}"
                    with st.expander(label):
                        st.markdown(f"**📖 Reference Answer:**\n> {res['reference']}")
                        st.markdown("---")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(f"**🤖 Gemma2-9b**  {score_badge(res['naive_llm_score'])}")
                            st.caption(res.get("naive_llm_reason", ""))
                        with col_b:
                            st.markdown(f"**📄 Mixtral RAG**  {score_badge(res['naive_rag_score'])}")
                            st.caption(res.get("naive_rag_reason", ""))
                        with col_c:
                            st.markdown(f"**⚖️ Aware RAG**  {score_badge(res['aware_score'])}")
                            st.caption(res.get("aware_reason", ""))

            except Exception as e:
                st.error(f"Evaluation Failed: {str(e)}")

        st.markdown("---")
        st.subheader("🔬 RAGAS IEEE-Standard Metrics")
        st.write("""
        **RAGAS** is the gold-standard framework for RAG evaluation used in IEEE/ACL/NeurIPS papers.
        It measures 4 mathematically grounded metrics on your actual retrieved document chunks —
        no keyword matching, no model opinion.
        """)
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info("📐 **Faithfulness** — Does the answer only use retrieved content?\n\n📎 **Answer Relevancy** — Does the answer actually address the question?")
        with col_info2:
            st.info("🎯 **Context Precision** — Are the retrieved chunks relevant? (signal:noise)\n\n🔁 **Context Recall** — Does the context cover the full reference answer?")

        n_q = st.slider("Number of questions to run RAGAS on", min_value=5, max_value=50, value=20, step=5)
        if st.button(f"🔬 Run RAGAS Analysis ({n_q} Questions)", use_container_width=True):
            try:
                rg_bar  = st.progress(0)
                rg_text = st.empty()

                def rg_progress(p, msg):
                    rg_bar.progress(p)
                    rg_text.text(msg)

                rg = run_ragas_evaluation(rg_progress, n_questions=n_q)
                rg_text.text("✅ RAGAS Complete!")

                nr = rg["naive_rag"]
                aw = rg["aware_rag"]
                imp = rg["improvement"]

                st.markdown("#### 📊 RAGAS Scores  *(0.0 – 1.0, higher is better)*")
                metric_names = {
                    "faithfulness":      "📐 Faithfulness",
                    "answer_relevancy":  "📎 Answer Relevancy",
                    "context_precision": "🎯 Context Precision",
                    "context_recall":    "🔁 Context Recall",
                    "ragas_score":       "⭐ RAGAS Score (avg)",
                }
                header_cols = st.columns([3, 2, 2, 2])
                header_cols[0].markdown("**Metric**")
                header_cols[1].markdown("**Naive RAG**")
                header_cols[2].markdown("**Aware RAG**")
                header_cols[3].markdown("**Improvement**")
                st.divider()

                for key, label in metric_names.items():
                    nv = nr[key]
                    av = aw[key]
                    diff = imp[key]
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                    c1.write(f"**{label}**")
                    c2.metric("", f"{nv:.3f}")
                    c3.metric("", f"{av:.3f}", delta=f"{diff:+.3f}")
                    arrow = "✅" if diff > 0 else "⚠️"
                    c4.write(f"{arrow} {'+' if diff>0 else ''}{diff:.3f}")

                st.markdown("#### 🔍 Per-Question RAGAS Scores")
                for i, (q, nq, aq) in enumerate(zip(
                    rg["questions"], nr["per_question"], aw["per_question"]
                )):
                    with st.expander(f"Q{i+1}: {q[:80]}..."):
                        cols = st.columns(5)
                        cols[0].markdown("**Metric**")
                        cols[1].markdown("Naive")
                        cols[2].markdown("Aware")
                        cols[3].markdown("Δ")
                        cols[4].markdown("Status")
                        for metric_key in ["faithfulness","answer_relevancy","context_precision","context_recall"]:
                            nv = nq.get(metric_key, 0) or 0
                            av = aq.get(metric_key, 0) or 0
                            d  = av - nv
                            c0, c1, c2, c3, c4 = st.columns(5)
                            c0.caption(metric_key.replace("_"," ").title())
                            c1.caption(f"{nv:.2f}")
                            c2.caption(f"{av:.2f}")
                            c3.caption(f"{d:+.2f}")
                            c4.caption("✅" if d >= 0 else "🔴")

            except Exception as e:
                st.error(f"RAGAS Failed: {str(e)}")
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
        pdf_files = []  # Initialize early to avoid scope issues in Rename section
        if os.path.exists("data/raw"):
            files = os.listdir("data/raw")
            pdf_files = [f for f in files if f.endswith(".pdf")]
            st.metric("Total Raw Documents", len(pdf_files))
            with st.expander("View Document Index"):
                for pdf in sorted(pdf_files):
                    prefix = "🔵" if pdf.startswith("base_") else "🟡" if pdf.startswith("amendment_") else "📄"
                    st.markdown(f"- {prefix} `{pdf}`")
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
