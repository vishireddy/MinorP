# Author Response & Manuscript Revisions for Reviewer Comment 1

**Paper Title:** LexPulse: An Adaptive Relationship-Aware RAG System for E-Governance Legal Policy Retrieval  
**Target Manuscript:** `editing_confpaper.pdf`  
**Target Comment:** Reviewer Comment 1 (Graph Models vs. JSON Metadata Contradiction)

---

## 1. Official Rebuttal Response (For Response Letter)

> **Reviewer Comment 1:**  
> *The paper is claimed to build a Policy Amendment Knowledge Graph, but it is stated by the methodology later that graph models are not used and relationships are stored mainly as JSON metadata, so this contradiction should be clearly resolved by the authors and the actual graph structure, nodes, edges, and traversal method should be explained.*

### **Author Response:**
We thank the reviewer for calling attention to the confusing and contradictory phrasing in Section VII-D. We clarify that LexPulse does construct and traverse a formal **Policy Amendment Relationship Graph** $\mathcal{G} = (V, E)$. However, rather than deploying a heavy external graph database engine (such as Neo4j or RDF triple stores), LexPulse represents this graph as a lightweight **JSON-serialized adjacency list** stored directly inside vector metadata payloads in ChromaDB.

In the revised manuscript (`editing_confpaper.pdf`), we have made the following specific updates:
1. **Removed the Contradiction:** Deleted the sentence in Section VII-D stating that *"this methodology does not utilize graph models"*.
2. **Added Formal Graph Notation:** Added mathematical definitions in Section VII-G for the graph topology $\mathcal{G} = (V, E, T_E)$, node attributes, edge types, and traversal mechanics.
3. **Formalized the Traversal Algorithm:** Derived the 1-Hop Adjacency Traversal equation used during context expansion.
4. **Unified Terminology:** Standardized terms across Section VI-E, Section VII-D, Section VII-G, Section VII-M (Working Algorithm), Section VIII-B, and Figures 1 and 3.

---

## 2. Exact Manuscript Changes (`editing_confpaper.pdf`)

### A. Fix Contradiction in Section VII-D (Page 5, Left Column, Lines 7–10)
* **Delete:**
  ```text
  The metadata will be saved in json file format. But this methodology does not utilize graph models to represent relationships between the documents.
  ```
* **Replace With:**
  ```text
  The extracted document relationships are modeled as a directed adjacency-list graph and serialized in JSON format alongside vector embeddings in ChromaDB payload metadata. This lightweight graph representation preserves node-edge dependency semantics while eliminating the compute overhead of external graph database engines.
  ```

---

### B. Formalize Graph Structure in Section VII-G (Page 5, Left Column)
* **Replace Section VII-G with:**
  ```text
  G. Metadata-Based Relationship Resolver Module

  Once document ingestion is complete, LexPulse models statutory relationships as a directed graph G = (V, E, T_E):
  1. Node Set (V): Document chunks v_i in V, where each node contains attributes <DocID_i, Status_i, Type_i>, with Status_i in {Active, Inactive, Superseded}.
  2. Edge Set (E) and Types (T_E): Directed dependency links e_ij = (v_i, v_j) in E representing legal relationships t in {AMENDS, SUPERSEDES, REFERENCES}.
  3. Traversal Method (1-Hop Adjacency Expansion): Given top-k retrieved chunks V_seed from hybrid search, the resolver expands context via:
     C_final = V_seed U ( U_{v_i in V_seed} { v_j in V | (v_i, v_j) in E OR (v_j, v_i) in E } )
  This ensures active amendments are automatically pulled into context whenever a superseded provision is retrieved.
  ```

---

### C. Update Section VI-E (Page 4, Left Column, Lines 40–42)
* **Delete:**
  ```text
  This architecture implements an economical disentanglement based communication mechanism to establish connections between related legislative documents in a structured Knowledge Graph.
  ```
* **Replace With:**
  ```text
  This architecture implements a lightweight metadata-driven relationship resolver to establish directed parent-amendment dependency edges between related legislative documents in a structured Policy Relationship Graph.
  ```

---

### D. Update Section VII-M Algorithm Steps (Page 6, Left Column)
* **Step 4:**
  `4. Perform metadata extraction to build the directed document relationship graph (serialized as JSON adjacency metadata).`
* **Step 11:**
  `11. Execute 1-hop graph traversal over the JSON adjacency lists to identify parent acts and active amendments.`
* **Step 12:**
  `12. Fetch and merge graph-traversed document nodes into the context window.`

---

### E. Update Figure Captions & Labels
* **Figure 1 (Page 4):** Relabel `Relationship Knowledge Graph` to `Policy Relationship Graph (JSON Adjacency Metadata)`.
* **Figure 3 (Page 6):** Keep header `Relationship Graph` and subtext `Raw JSON Topology of Policy Nodes`.

---

## 3. Page Count Impact Verification
- **Total Lines Removed:** ~5 lines
- **Total Lines Added:** ~8 lines
- **Net Growth:** **+3 lines total** across the 8-page paper.
- **Page Limit Status:** Fits comfortably within existing whitespace on Pages 5 and 6 without pushing text to an 9th page.
