# Complete Code Execution Flow & Architecture Guide
**Relationship-Aware Legal RAG System**

This document breaks down the exact execution flow of the python scripts in your repository. It explains how data moves from a raw PDF document all the way into the final AI response.

---

## 1. Directory Structure
```text
MinorP/
├── app.py                      # The Frontend UI (Streamlit)
├── data/
│   ├── raw/                    # Raw legal PDF documents
│   ├── chroma_db/              # Persistent Vector Database
│   └── relationship_graph.json # Autonomous Knowledge Graph
├── src/
│   ├── ingestion.py          # Script 1: PDF Splitting
│   ├── metadata_tagger.py    # Script 2: Graph Builder
│   ├── retrieval_engine.py   # Script 3: Hybrid Search & Injector
│   └── evaluate.py           # Script 4: AI Testing Suite
└── requirements.txt
```

---

## 2. Ingestion & Graph Building Flow
*Triggered when you click "Run Indexing Pipeline" in Streamlit.*

### Step A: Data Chunking (`src/ingestion.py`)
- **Execution:** The application calls `load_and_chunk_pdfs()`.
- **Purpose:** Vector databases cannot read 100-page books at once. This script uses LangChain's `PyPDFDirectoryLoader` to read every PDF in `data/raw`.
- **Action:** It runs a `RecursiveCharacterTextSplitter` which takes the 100+ pages and slices them into tiny blocks of 500 characters. Crucially, it leaves a "10% overlap" between blocks so that sentences aren't cut in half.

### Step B: The Knowledge Graph (`src/metadata_tagger.py`)
- **Execution:** Immediately after chunking, the app calls `update_relationship_graph(docs)`.
- **Purpose:** To autonomously map out which base policies have been superseded by amendments.
- **Action:** Instead of reading the whole document, the script extracts just the **first 1000 characters** (the preamble/title page) of every unique PDF. 
- **AI Extraction:** It sends those preambles to the Groq AI (`llama-3.1-8b-instant`) with a strict programmatic prompt. The AI acts as a parser, analyzing the legal titles and returning pure JSON data identifying `amended_by: []`.
- **Storage:** The output is saved physically to `data/relationship_graph.json`.

### Step C: Vector Database Storage
- **Action:** The 500-character chunks are embedded using HuggingFace `all-MiniLM-L6-v2` (converting English into 384-dimensional math coordinate vectors). These math coordinates are permanently saved to `data/chroma_db`.

---

## 3. The Retrieval & Answer Flow
*Triggered when a user types a question in the Streamlit search bar.*

### Step A: Hybrid Ensemble Search (`src/retrieval_engine.py`)
- **Execution:** The user's text question is sent to `create_relationship_aware_rag_chain()`.
- **Mechanism:** To avoid "Context Dilution" (where large PDFs overpower small ones), the query is run through two search engines simultaneously:
  1. **ChromaDB Vector Search:** Does math on the meaning of words.
  2. **BM25 Lexical Search:** Does exact keyword matching (like finding the exact word "Commissioner").
- **Reciprocal Rank Fusion (RRF):** A custom python class named `HybridRRFRetriever` scores the results from both engines and returns the Top 6 absolute best text blocks.

### Step B: The Python Intercept Wrapper (The Secret Sauce)
- **Execution:** Inside `src/retrieval_engine.py`, the `RAGWrapper.invoke()` function intercepts the Top 6 blocks before the AI is allowed to see them.
- **Mechanism:**
  1. The wrapper looks at the filenames of the 6 blocks (e.g., `base_policy.pdf`).
  2. It opens `relationship_graph.json`.
  3. It spots a rule: *"base_policy.pdf was amended_by amendment_policy.pdf"*.
  4. It checks if `amendment_policy.pdf` is in the Top 6 blocks. If it isn't (due to the semantic gap), the wrapper **physically forces ChromaDB to open**, extracts 3 blocks from `amendment_policy.pdf`, and forcibly staples them to the context list.

### Step C: Zero-Hallucination AI Generation
- **Mechanism:** The Python Intercept Wrapper formats the text strings to feed to Groq. 
- **Action:** It dynamically writes a warning label on the text depending on the Knowledge Graph. For example, it prefixes `base_policy` text with `⚠️ SUPERSEDED BY: amendment_policy.pdf`.
- **Execution:** The modified, tagged text package is finally sent to Groq AI. Because of the warning labels, the AI is mathematically forced to write its final response chronologically (e.g. "Initially it was X, but it was amended to Y.").

---

## 4. Evaluation Suite (`src/evaluate.py`)
- **Purpose:** This is purely a testing bench meant to prove the system works to your professors.
- **Action:** It contains a hardcoded `Naive RAG` pipeline (which acts like a basic, un-modified ChatGPT) and our custom `Relationship-Aware` pipeline.
- It runs the same question through both systems side-by-side, definitively proving that the Naive RAG provides an illegal, outdated answer, while our system perfectly bridges the semantic gap.
