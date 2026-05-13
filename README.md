# LexPulse ⚖️
### Adaptive Legal Policy Intelligence

> **LexPulse: Mitigating Amendment Blindness in E-Governance Legal QA using Adaptive Relationship-Aware RAG**

A research-grade AI platform that accurately answers questions about Indian legal policy documents — including laws that have been amended or superseded — using a custom **Relationship-Aware Retrieval-Augmented Generation (RAG)** pipeline with an **Adaptive LLM Fallback** to eliminate the RAG Penalty.

---

## 🏆 Key Results (LLM-as-a-Judge · 51 Legal Questions)

| Pipeline | Overall Accuracy | Amendment-Trap Accuracy |
|---|---|---|
| Base LLM (No RAG) | 54.9% | 55.6% |
| Naive RAG | 54.9% | 44.4% ⚠️ |
| **Adaptive RAG (LexPulse)** | **70.6% ✅** | **72.2% ✅** |

> LexPulse achieves a **+15.7 percentage point** improvement over Naive RAG on Amendment-Trap queries, directly proving the core thesis: **Adaptive Relationship-Aware RAG cures Amendment Blindness.**

---

## 🧠 What Makes LexPulse Different

Standard RAG systems suffer from **"Amendment Blindness"** — they retrieve the original version of a law while completely missing the amendment that superseded it. LexPulse solves this with two innovations:

1. **Policy Amendment Knowledge Graph** (`data/relationship_graph.json`)
   - A JSON graph where nodes = Acts, edges = `amended_by` relationships
   - At query time, the retriever **traverses the graph** to automatically inject amendment documents that vector search missed

2. **Adaptive LLM Fallback**
   - When the vector database has no relevant context, instead of hallucinating, LexPulse gracefully falls back to the LLM's internal legal knowledge
   - This eliminates the "RAG Penalty" — the accuracy drop that Naive RAG suffers on general questions

---

## 🗂️ Project Structure

```
LexPulse/
├── server.py                  # Custom HTTP server (API backend)
├── frontend/
│   └── index.html             # Web dashboard UI
├── src/
│   ├── retrieval_engine.py    # Adaptive RAG + Knowledge Graph injection
│   ├── evaluate.py            # LLM-as-a-Judge evaluation suite (51 Qs)
│   ├── ingestion.py           # PDF chunking + metadata enrichment
│   ├── metadata_tagger.py     # Amendment relationship tagger
│   ├── results_manager.py     # Eval results persistence
│   └── auth_db.py             # User authentication
├── data/
│   ├── relationship_graph.json  # Policy Amendment Knowledge Graph
│   ├── eval_results_v2.json     # Latest evaluation results
│   └── raw/                     # Source PDF directory (PDFs not included)
├── generate_eval_chart.py     # Publication-quality results chart generator
├── requirements.txt
└── .env.example               # API key template
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/LexPulse.git
cd LexPulse
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or: .venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```
Get a free API key at [console.groq.com](https://console.groq.com)

### 5. Add your PDF documents
Place Indian legal policy PDFs in `data/raw/`. Then run System Sync from the dashboard to build the vector database.

### 6. Start the server
```bash
python server.py
```
Open [http://localhost:8080](http://localhost:8080) in your browser.

---

## 📊 Running the Evaluation Suite

From the dashboard, navigate to **Evaluation Lab** → click **Run Suite**.

Or from the terminal:
```bash
python -m src.evaluate
```

Results are saved to `data/eval_results_v2.json` and reflected live in the dashboard.

---

## 📄 Paper

**Title:** *LexPulse: Mitigating Amendment Blindness in E-Governance Legal QA using Adaptive Relationship-Aware RAG*

Submitted to: [Conference Name]

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA-3.1-8b-Instant via Groq API |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Hybrid Retrieval | BM25 + Semantic Search |
| Knowledge Graph | Custom JSON graph (`relationship_graph.json`) |
| Backend | Python `http.server` |
| Frontend | Vanilla HTML/CSS/JS + TailwindCSS |

---

## 👨‍💻 Authors

- Vishwak Reddy
- Pranathi Chalamalasetti
- Perabathini Varshini
- Pasulavadi Harini

---

## 📜 License

MIT License — free to use for academic and research purposes.
