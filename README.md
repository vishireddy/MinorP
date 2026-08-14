# LexPulse ⚖️
### Adaptive Legal Policy Intelligence

> **LexPulse: Mitigating Amendment Blindness in E-Governance Legal QA using Adaptive Relationship-Aware RAG**

A research-grade AI platform that answers questions about Indian legal policy documents — including laws that have been amended or superseded — using a custom **Relationship-Aware Retrieval-Augmented Generation (RAG)** pipeline with an **Adaptive LLM Fallback** to reduce the typical RAG accuracy penalty.

---

## 🏆 Key Results (LLM-as-a-Judge · 51 Legal Questions)

| Pipeline | Overall Accuracy | Amendment-Trap Accuracy |
|---|---|---|
| Base LLM (No RAG) | 54.9% | 55.6% |
| Naive RAG | 54.9% | 44.4% ⚠️ |
| **Adaptive RAG (LexPulse)** | **70.6% ✅** | **72.2% ✅** |

> LexPulse achieves a **+15.7 percentage point** overall accuracy improvement over Naive RAG, and a **+27.8 percentage point** improvement on Amendment-Trap queries, based on the current 51-question internal evaluation. Broader, independent validation remains future work.

---

## 🧠 What Makes LexPulse Different

Standard RAG systems suffer from **"Amendment Blindness"** — they retrieve the original version of a law while completely missing the amendment that superseded it. LexPulse solves this with two innovations:

1. **JSON-encoded Policy Amendment Relationship Graph** (`data/relationship_graph.json`)
   - A JSON graph where nodes = documents and edges encode amendment/supersession relationships
   - At query time, the retriever uses this graph metadata to inject active amendment documents that vector search alone may miss

### Implementation details — Relationship Graph (JSON metadata)

The Policy Amendment Relationship Graph is encoded as JSON metadata rather than stored in a graph-database engine. Each node corresponds to a document identifier (file name or canonical statutory ID). Directed edges are represented via the fields `amends` (amendment → base document) and `amended_by` (base → list of later amendments). Metadata also records `is_amendment`, `status` (active | inactive | superseded), and `effective_date` when available. The graph is produced by an LLM-assisted metadata extractor followed by deterministic token/identifier matching and curated manual overrides; the builder is implemented in `src/metadata_tagger.py` and persisted to `data/relationship_graph.json`.

At query time, hybrid BM25 + vector retrieval produces candidate chunks. For any retrieved base document that has active `amended_by` links missing from the candidate set, the pipeline performs a targeted similarity search constrained to the amendment's source and injects the top-k amendment chunks into the final LLM context. This one-hop, metadata-driven injection makes retrieval relationship-aware without requiring a graph database or multi-hop graph traversal engine.

Pseudocode (one-hop injection):

```python
def inject_amendments(candidates, graph, vectorstore, k=3):
   augmented = list(candidates)
   candidate_sources = {c['source'] for c in candidates}
   for c in candidates:
      doc_id = c['source']
      for amend_id in graph.get(doc_id, {}).get('amended_by', []):
         if amend_id not in candidate_sources:
            amendment_chunks = vectorstore.similarity_search_by_source(amend_id, top_k=k)
            augmented.extend(amendment_chunks)
            candidate_sources.update(ch['source'] for ch in amendment_chunks)
   return augmented
```

2. **Adaptive LLM Fallback**
   - When the vector database has no relevant context, LexPulse gracefully falls back to the LLM's internal legal knowledge
   - This helps reduce the typical RAG accuracy penalty found in some baseline deployments

---

## 🗂️ Project Structure

```
LexPulse/
├── server.py                  # Custom HTTP server (API backend)
├── frontend/
│   └── index.html             # Web dashboard UI
├── src/
│   ├── retrieval_engine.py    # Adaptive RAG + relationship graph injection
│   ├── evaluate.py            # LLM-as-a-Judge evaluation suite (51 Qs)
│   ├── ingestion.py           # PDF chunking + metadata enrichment
│   ├── metadata_tagger.py     # Amendment relationship tagger
│   ├── results_manager.py     # Eval results persistence
│   └── auth_db.py             # User authentication
├── data/
│   ├── relationship_graph.json  # Policy Amendment Relationship Graph
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
| Relationship Graph | Custom JSON graph (`relationship_graph.json`) |
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
