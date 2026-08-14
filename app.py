import streamlit as st
import os
from dotenv import load_dotenv
from src.auth_db import init_db, register_user, verify_user, save_chat, get_chat_history, purge_system_chats
from src.evaluate import run_evaluation_suite, run_ragas_evaluation, run_ablation_study
from src.ingestion import load_and_chunk_pdfs
from src.metadata_tagger import enrich_metadata
from src.retrieval_engine import get_vectorstore, create_relationship_aware_rag_chain
from src.api_ingestion import download_act_pdf, search_acts, get_available_acts, fetch_from_any_url
from src.results_manager import (
    load_eval_results, load_ablation_results, load_ragas_results,
    results_exist, ablation_results_exist, ragas_results_exist,
    export_breakdown_to_csv_bytes, export_metrics_to_csv_bytes, generate_paper_stats
)

load_dotenv()
init_db()

st.set_page_config(page_title="Relationship-Aware RAG", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Modern Buttons */
.stButton>button, .stDownloadButton>button {
    border-radius: 8px !important;
    border: none !important;
    background: #2563eb !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2), 0 2px 4px -1px rgba(37, 99, 235, 0.1) !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3), 0 4px 6px -2px rgba(37, 99, 235, 0.15) !important;
}

/* Chat Messages */
div[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    background-color: #ffffff !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
    padding: 1.2rem !important;
}
div[data-testid="stChatMessage"] * {
    color: #1e293b !important;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s ease-in-out !important;
}
div[data-testid="metric-container"]:hover {
    border-color: #cbd5e1 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    transform: translateY(-2px) !important;
}
div[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 600 !important;
}

/* Inputs */
.stTextInput>div>div>input {
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
    padding: 0.5rem 1rem !important;
}
.stTextInput>div>div>input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
}

/* Expander */
.streamlit-expanderHeader {
    border-radius: 8px !important;
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* Google Translate Styling */
.goog-te-gadget-simple {
    border: none !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
}
.goog-te-gadget-icon {
    display: none !important;
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
    
    st.markdown("### ⚙️ Engine Settings")
    st.session_state['strict_mode'] = st.toggle("Strict RAG Mode (No LLM Fallback)", value=False, help="When ON, the AI will refuse to answer if the context is missing. When OFF, it falls back to internal knowledge.")

# ==========================================
# 2. MAIN HEADER
# ==========================================
import streamlit.components.v1 as components
components.html("""
    <div id="google_translate_element"></div>
    <script type="text/javascript">
        function googleTranslateElementInit() {
            new google.translate.TranslateElement({pageLanguage: 'en', layout: google.translate.TranslateElement.InlineLayout.SIMPLE}, 'google_translate_element');
        }
    </script>
    <script type="text/javascript" src="//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
    <style>
        body { margin: 0; padding: 0; font-family: sans-serif; }
        .goog-te-gadget-simple { border: 1px solid #e2e8f0 !important; border-radius: 4px !important; padding: 5px !important; }
    </style>
""", height=50)

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
                        is_strict = st.session_state.get('strict_mode', False)
                        rag_chain = create_relationship_aware_rag_chain(strict_mode=is_strict)
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
    tab_sync, tab_graph, tab_eval, tab_ablation, tab_manage = st.tabs(
        ["🚀 System Sync", "🕸️ Relationship Graph", "📊 Evaluation Lab", "🧪 Ablation Study", "🗄️ File Manager"]
    )
    
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
        st.subheader("📊 RAG Evaluation Suite")

        # ── Cached results loader ──────────────────────────────
        if results_exist():
            cached = load_eval_results()
            ts = cached.get("timestamp", "Unknown") if cached else "Unknown"
            st.success(f"✅ Cached results found (run: {ts}). Load instantly or re-run below.")
            col_load, col_csv1, col_csv2 = st.columns(3)
            with col_load:
                if st.button("📂 Load Cached Results", use_container_width=True):
                    st.session_state["eval_results"] = cached
                    st.rerun()
            if cached:
                with col_csv1:
                    st.download_button("⬇️ Export Breakdown", data=export_breakdown_to_csv_bytes(cached),
                                       file_name="eval_breakdown.csv", mime="text/csv", use_container_width=True)
                with col_csv2:
                    st.download_button("⬇️ Export Metrics", data=export_metrics_to_csv_bytes(cached),
                                       file_name="eval_metrics.csv", mime="text/csv", use_container_width=True)

        if st.button("▶️ Run Full Evaluation (50 Questions)", use_container_width=True, type="primary"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                def update_progress(p, m):
                    progress_bar.progress(p)
                    status_text.text(m)
                results = run_evaluation_suite(update_progress)
                status_text.text("✅ Evaluation Complete! Results auto-saved.")
                st.session_state["eval_results"] = {"results": results}
                st.rerun()
            except Exception as e:
                st.error(f"Evaluation Failed: {str(e)}")

        # ── Display results (from session or cached) ───────────
        display_results = st.session_state.get("eval_results")
        if display_results:
            results = display_results.get("results", display_results)
            m = results["metrics"]

            st.markdown("---")
            st.subheader("Overall Performance  *(Pass = Score ≥ 6/10)*")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🤖 No RAG", f"{m['naive_llm_accuracy']:.1f}%")
            col2.metric("📄 Naive RAG", f"{m['naive_rag_accuracy']:.1f}%",
                        delta=f"{m['naive_rag_accuracy'] - m['naive_llm_accuracy']:+.1f}% vs No RAG")
            col3.metric("⚖️ Aware RAG", f"{m['aware_accuracy']:.1f}%",
                        delta=f"{m['rag_improvement_over_llm']:+.1f}% vs No RAG")
            col4.metric("Test Size", f"{m['total_queries']} Qs")

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("🤖 Avg Judge Score (No RAG)", f"{m['naive_llm_avg_score']:.2f}/10")
            c2.metric("📄 Avg Judge Score (Naive RAG)", f"{m['naive_rag_avg_score']:.2f}/10",
                      delta=f"{m['naive_rag_avg_score'] - m['naive_llm_avg_score']:+.2f}")
            c3.metric("⚖️ Avg Judge Score (Aware RAG)", f"{m['aware_avg_score']:.2f}/10",
                      delta=f"{m['aware_avg_score'] - m['naive_llm_avg_score']:+.2f}")

            # Latency
            nl_lat = m.get("naive_llm_avg_latency_ms")
            nr_lat = m.get("naive_rag_avg_latency_ms")
            aw_lat = m.get("aware_avg_latency_ms")
            if nl_lat and nr_lat and aw_lat:
                st.markdown("<br>", unsafe_allow_html=True)
                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("⏱️ Latency (No RAG)", f"{nl_lat} ms")
                lc2.metric("⏱️ Latency (Naive RAG)", f"{nr_lat} ms", delta=f"{nr_lat - nl_lat:+d} ms")
                lc3.metric("⏱️ Latency (Aware RAG)", f"{aw_lat} ms", delta=f"{aw_lat - nl_lat:+d} ms")

            st.markdown("---")
            st.subheader("🎯 Amendment-Trap Pass Rates")
            c1, c2, c3 = st.columns(3)
            c1.metric("🤖 No RAG", f"{m['tricky_naive_llm_accuracy']:.1f}%")
            c2.metric("📄 Naive RAG", f"{m['tricky_naive_rag_accuracy']:.1f}%",
                      delta=f"{m['tricky_naive_rag_accuracy'] - m['tricky_naive_llm_accuracy']:+.1f}%")
            c3.metric("⚖️ Aware RAG", f"{m['tricky_aware_accuracy']:.1f}%",
                      delta=f"{m['tricky_aware_accuracy'] - m['tricky_naive_llm_accuracy']:+.1f}%")

            st.markdown("---")
            with st.expander("📂 View Category Breakdown"):
                for cat, s in results["category_scores"].items():
                    t = s["total"]
                    nl = s["naive_llm_pass"] / t * 100
                    nr = s["naive_rag_pass"] / t * 100
                    aw = s["aware_pass"] / t * 100
                    c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
                    c1.write(f"**{cat}**")
                    c2.metric("No RAG", f"{nl:.0f}%")
                    c3.metric("Naive RAG", f"{nr:.0f}%", delta=f"{nr-nl:+.0f}%")
                    c4.metric("Aware RAG", f"{aw:.0f}%", delta=f"{aw-nl:+.0f}%")
                    c5.caption(f"{t} Qs")

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🔍 Question-by-Question Analysis")
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
                        st.markdown(f"**🤖 LLaMA3.3 No RAG**  {score_badge(res['naive_llm_score'])}")
                        st.caption(res.get("naive_llm_reason", ""))
                        if res.get("naive_llm_latency_ms"):
                            st.caption(f"⏱ {res['naive_llm_latency_ms']} ms")
                    with col_b:
                        st.markdown(f"**📄 Naive RAG**  {score_badge(res['naive_rag_score'])}")
                        st.caption(res.get("naive_rag_reason", ""))
                        if res.get("naive_rag_latency_ms"):
                            st.caption(f"⏱ {res['naive_rag_latency_ms']} ms")
                    with col_c:
                        st.markdown(f"**⚖️ Aware RAG**  {score_badge(res['aware_score'])}")
                        st.caption(res.get("aware_reason", ""))
                        if res.get("aware_latency_ms"):
                            st.caption(f"⏱ {res['aware_latency_ms']} ms")



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
    with tab_ablation:
        st.write("""
        **Ablation Study** — isolates the exact contribution of the Relationship Graph.

        Both pipelines use identical **Hybrid BM25 + Vector (RRF)** retrieval and **LLaMA3.1-8b**.
        The only difference:
        - **Pipeline A (Hybrid RAG Only):** Relationship injection DISABLED
        - **Pipeline B (Aware RAG):** Relationship injection ENABLED ✅

        The delta between them proves the graph's value beyond retrieval alone.
        """)

        if ablation_results_exist():
            abl_cached = load_ablation_results()
            abl_ts = abl_cached.get("timestamp", "Unknown") if abl_cached else "Unknown"
            st.success(f"✅ Cached ablation results found (run: {abl_ts}).")
            if st.button("📂 Load Cached Ablation Results", use_container_width=True):
                st.session_state["ablation_results"] = abl_cached
                st.rerun()

        n_abl = st.slider("Questions for ablation", min_value=10, max_value=50, value=50, step=10)
        if st.button(f"▶️ Run Ablation Study ({n_abl} Questions)", use_container_width=True, type="primary"):
            try:
                abl_bar  = st.progress(0)
                abl_text = st.empty()
                def abl_progress(p, msg):
                    abl_bar.progress(p)
                    abl_text.text(msg)
                abl_res = run_ablation_study(abl_progress, n_questions=n_abl)
                abl_text.text("✅ Ablation Complete! Results auto-saved.")
                st.session_state["ablation_results"] = abl_res
                st.rerun()
            except Exception as e:
                st.error(f"Ablation Failed: {str(e)}")

        abl_display = st.session_state.get("ablation_results")
        if abl_display:
            am = abl_display.get("metrics", abl_display.get("results", {}).get("metrics", {}))

            st.markdown("---")
            st.subheader("📊 Ablation Results — 4 Pipelines")
            st.caption("Progressive component addition: each row adds exactly one architectural element.")

            # Pass rates — 4 columns
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🤖 No RAG\n(LLaMA3.3-70b)", f"{am.get('no_rag_pass_rate', 0):.1f}%",
                      help="Pure LLM — no documents retrieved")
            c2.metric("📄 Naive RAG\n(Qwen3-32b)", f"{am.get('naive_rag_pass_rate', 0):.1f}%",
                      delta=f"{am.get('retrieval_gain', 0):+.1f}% vs No RAG",
                      help="Basic vector retrieval added")
            c3.metric("🔗 Hybrid RAG\n(LLaMA3.1-8b)", f"{am.get('hybrid_pass_rate', 0):.1f}%",
                      delta=f"{am.get('hybrid_gain', 0):+.1f}% vs Naive RAG",
                      help="BM25 + Vector RRF added, no graph")
            c4.metric("⚖️ Aware RAG ✅\n(LLaMA3.1-8b)", f"{am.get('aware_pass_rate', 0):.1f}%",
                      delta=f"{am.get('graph_gain', 0):+.1f}% vs Hybrid",
                      help="Relationship graph injection added")

            st.markdown("#### 🏅 Average Judge Score (/10)")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("No RAG",    f"{am.get('no_rag_avg_score', 0):.2f}")
            s2.metric("Naive RAG", f"{am.get('naive_rag_avg_score', 0):.2f}",
                      delta=f"{am.get('naive_rag_avg_score', 0) - am.get('no_rag_avg_score', 0):+.2f}")
            s3.metric("Hybrid RAG", f"{am.get('hybrid_avg_score', 0):.2f}",
                      delta=f"{am.get('hybrid_avg_score', 0) - am.get('naive_rag_avg_score', 0):+.2f}")
            s4.metric("Aware RAG", f"{am.get('aware_avg_score', 0):.2f}",
                      delta=f"{am.get('aware_avg_score', 0) - am.get('hybrid_avg_score', 0):+.2f}")

            if am.get("aware_avg_latency"):
                st.markdown("#### ⏱️ Avg Response Latency (ms)")
                l1, l2, l3, l4 = st.columns(4)
                l1.metric("No RAG",    f"{am.get('no_rag_avg_latency', 0)} ms")
                l2.metric("Naive RAG", f"{am.get('naive_rag_avg_latency', 0)} ms")
                l3.metric("Hybrid RAG", f"{am.get('hybrid_avg_latency', 0)} ms")
                l4.metric("Aware RAG", f"{am.get('aware_avg_latency', 0)} ms")

            st.markdown("#### 🎯 Amendment-Trap Pass Rate")
            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("Trap Questions", am.get("tricky_total", 0))
            t2.metric("No RAG",    f"{am.get('tricky_no_rag', 0):.1f}%")
            t3.metric("Naive RAG", f"{am.get('tricky_naive_rag', 0):.1f}%",
                      delta=f"{am.get('tricky_naive_rag', 0) - am.get('tricky_no_rag', 0):+.1f}%")
            t4.metric("Hybrid RAG", f"{am.get('tricky_hybrid', 0):.1f}%",
                      delta=f"{am.get('tricky_hybrid', 0) - am.get('tricky_naive_rag', 0):+.1f}%")
            t5.metric("Aware RAG", f"{am.get('tricky_aware', 0):.1f}%",
                      delta=f"{am.get('tricky_aware', 0) - am.get('tricky_hybrid', 0):+.1f}%")

            st.markdown("---")
            st.subheader("🔍 Question-by-Question Ablation Breakdown")
            for i, r in enumerate(abl_display.get("breakdown", [])):
                trap_tag = " 🎯" if r.get("tricky") else ""
                with st.expander(f"Q{i+1}.{trap_tag} [{r['category']}] {r['query']}"):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        badge = "🟢" if r.get("no_rag_pass") else "🔴"
                        st.markdown(f"**{badge} No RAG**  {r.get('no_rag_score', 0)}/10")

                        st.caption(r.get("no_rag_reason", ""))
                        st.caption(f"⏱ {r.get('no_rag_latency_ms', 0)} ms")
                    with col_b:
                        badge = "🟢" if r.get("naive_rag_pass") else "🔴"
                        st.markdown(f"**{badge} Naive RAG**  {r.get('naive_rag_score', 0)}/10")
                        st.caption(r.get("naive_rag_reason", ""))
                        st.caption(f"⏱ {r.get('naive_rag_latency_ms', 0)} ms")
                    with col_c:
                        badge = "🟢" if r.get("hybrid_pass") else "🔴"
                        st.markdown(f"**{badge} Hybrid RAG**  {r.get('hybrid_score', 0)}/10")
                        st.caption(r.get("hybrid_reason", ""))
                        st.caption(f"⏱ {r.get('hybrid_latency_ms', 0)} ms")
                    with col_d:
                        badge = "🟢" if r.get("aware_pass") else "🔴"
                        st.markdown(f"**{badge} Aware RAG ✅**  {r.get('aware_score', 0)}/10")
                        st.caption(r.get("aware_reason", ""))
                        st.caption(f"⏱ {r.get('aware_latency_ms', 0)} ms")

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
