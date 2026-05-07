import os
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain_core.documents import Document

CHROMA_PATH = "data/chroma_db"

def get_vectorstore(chunks: List[Document] = None):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if chunks:
        print(f"Creating/Updating ChromaDB with {len(chunks)} chunks at {CHROMA_PATH}...")
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH
        )
    else:
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectorstore


def _build_hybrid_retriever(vectorstore):
    """Builds a Hybrid BM25+Vector RRF retriever from the vectorstore.
    Uses paginated fetching to avoid SQLite's 'too many SQL variables' limit.
    """
    from langchain_community.retrievers import BM25Retriever
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 8}
    )

    # Paginate in batches of 200 to stay within SQLite variable limits
    bm25_docs = []
    batch_size = 200
    offset = 0
    while True:
        try:
            batch = vectorstore.get(limit=batch_size, offset=offset,
                                    include=["documents", "metadatas"])
        except Exception:
            break
        if not batch or not batch.get("documents"):
            break
        for t, m in zip(batch["documents"], batch["metadatas"]):
            bm25_docs.append(Document(page_content=t, metadata=m))
        if len(batch["documents"]) < batch_size:
            break
        offset += batch_size

    if bm25_docs:
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = 8

        class HybridRRFRetriever:
            def __init__(self, bm25, vector, w_bm25=0.3, w_vec=0.7):
                self.bm25 = bm25
                self.vector = vector
                self.w_bm25 = w_bm25
                self.w_vec = w_vec

            def invoke(self, query):
                docs_bm25 = self.bm25.invoke(query)
                docs_vec = self.vector.invoke(query)
                rrf_scores = {}
                doc_map = {}
                k = 60
                for rank, d in enumerate(docs_bm25):
                    score = self.w_bm25 * (1.0 / (rank + k))
                    doc_map[d.page_content] = d
                    rrf_scores[d.page_content] = rrf_scores.get(d.page_content, 0.0) + score
                for rank, d in enumerate(docs_vec):
                    score = self.w_vec * (1.0 / (rank + k))
                    doc_map[d.page_content] = d
                    rrf_scores[d.page_content] = rrf_scores.get(d.page_content, 0.0) + score
                top = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
                return [doc_map[c] for c in top[:8]]

        return HybridRRFRetriever(bm25_retriever, vector_retriever)
    return vector_retriever


def _inject_relationships(docs, vectorstore):
    """
    Core relationship injection logic — pulls in amendment docs
    that the vector search missed (the semantic gap fix).
    Uses targeted similarity_search to avoid SQLite variable-limit errors.
    Returns (enriched_docs, graph) tuple.
    """
    import json
    graph_path = "data/relationship_graph.json"
    graph = {}
    if not os.path.exists(graph_path):
        return docs, graph
    try:
        with open(graph_path, "r") as f:
            graph = json.load(f)

        retrieved_files = set(
            os.path.basename(d.metadata.get("source", "")) for d in docs
        )
        injected_docs = []

        for fname in list(retrieved_files):
            if fname not in graph:
                continue
            for amendment_file in graph[fname].get("amended_by", []):
                if amendment_file in retrieved_files:
                    continue
                # Targeted fetch — use similarity_search with metadata filter
                # instead of vectorstore.get() to avoid SQLite variable limits
                amendment_stem = amendment_file.replace(".pdf", "").replace("_", " ")
                try:
                    candidates = vectorstore.similarity_search(
                        query=amendment_stem, k=3,
                        filter={"source": {"$contains": amendment_file}}
                    )
                except Exception:
                    # Fallback: search without filter if $contains unsupported
                    try:
                        candidates = vectorstore.similarity_search(
                            query=amendment_stem, k=5
                        )
                        # Keep only chunks whose source matches
                        candidates = [
                            d for d in candidates
                            if amendment_file in d.metadata.get("source", "")
                        ][:3]
                    except Exception:
                        candidates = []

                injected_docs.extend(candidates)
                retrieved_files.add(amendment_file)

        docs = list(docs) + injected_docs
    except Exception:
        pass
    return docs, graph


def _format_context_with_tags(docs, graph):
    """Adds relationship status tags to each chunk for the LLM."""
    formatted = []
    for d in docs:
        fname = os.path.basename(d.metadata.get("source", "Unknown"))
        status = graph.get(fname, {}).get("status", "Active")
        amended_by = graph.get(fname, {}).get("amended_by", [])
        chunk_str = f"📜 Document: {fname}\n⚠️ RELATIONSHIP STATUS: {status}\n"
        if amended_by:
            chunk_str += f"⚠️ SUPERSEDED BY: {', '.join(amended_by)}\n"
        chunk_str += f"Text:\n{d.page_content}"
        formatted.append(chunk_str)
    return "\n\n---\n\n".join(formatted)


STRICT_SYSTEM_PROMPT = (
    "You are a highly advanced E-Governance AI assistant with deep pre-trained knowledge of Indian Law.\n"
    "STRICT DOCUMENT GROUNDING: You must combine the exact facts from the provided policy context with your extensive "
    "pre-trained legal reasoning capabilities to formulate a highly accurate, comprehensive, and human-readable answer.\n"
    "CRITICAL INSTRUCTION: Pay VERY CLOSE ATTENTION to the relationship status of the documents.\n"
    "If you see a document tagged as 'Inactive/Superseded' and another tagged as 'Active' or 'amends',\n"
    "you MUST base your factual legal claims strictly on the Active amendment to prevent 'amendment blindness'.\n"
    "Always cite your sources from the context, providing the exact Document Name.\n"
    "If the Context is completely empty or irrelevant to the user's query, politely state that the information "
    "is not present in the active policy database. Do not use internal memory.\n\nContext:\n{context}"
)

ADAPTIVE_SYSTEM_PROMPT = (
    "You are a highly advanced E-Governance AI assistant with deep pre-trained knowledge of Indian Law.\n"
    "ADAPTIVE KNOWLEDGE FUSION: You must answer the user's query with maximum factual accuracy.\n"
    "Step 1: Read the provided retrieved context. Pay VERY CLOSE ATTENTION to relationship statuses (Active vs Superseded/Amends).\n"
    "If the context contains sufficient and relevant information to answer the query accurately, base your answer strictly on the context, "
    "explain policy updates clearly, and append the exact phrase '[Source: Verified Legal Database]' to the very end of your response.\n\n"
    "Step 2: If the retrieved context is completely empty, fragmented, irrelevant, or insufficient to provide a complete and accurate legal answer, "
    "you MUST abandon the context and fall back to your extensive internal pre-trained legal knowledge to formulate a highly accurate answer. "
    "If you use your internal memory, you MUST append the exact phrase '[Source: Internal LLM Memory]' to the very end of your response.\n\n"
    "Context:\n{context}"
)


_GLOBAL_RETRIEVER = None
_GLOBAL_VECTORSTORE = None

def create_relationship_aware_rag_chain(strict_mode=False):
    """Full pipeline: Hybrid retrieval + relationship injection + context tagging."""
    global _GLOBAL_RETRIEVER, _GLOBAL_VECTORSTORE
    
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError("ChromaDB not found. Please upload PDFs first.")

    from langchain_groq import ChatGroq
    from langchain_core.output_parsers import StrOutputParser

    if _GLOBAL_VECTORSTORE is None:
        _GLOBAL_VECTORSTORE = get_vectorstore()
    vectorstore = _GLOBAL_VECTORSTORE
    
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = _build_hybrid_retriever(vectorstore)
    retriever = _GLOBAL_RETRIEVER

    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    selected_prompt = STRICT_SYSTEM_PROMPT if strict_mode else ADAPTIVE_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", selected_prompt),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    class RAGWrapper:
        def invoke(self, inputs):
            query = inputs["input"]
            t0 = time.time()
            docs = list(retriever.invoke(query))
            docs, graph = _inject_relationships(docs, vectorstore)
            context = _format_context_with_tags(docs, graph)
            ans = chain.invoke({"context": context, "input": query})
            latency_ms = round((time.time() - t0) * 1000)
            return {"answer": ans, "context": docs, "latency_ms": latency_ms}

    return RAGWrapper()


def create_ablation_chain():
    """
    Ablation pipeline: Hybrid retrieval ONLY — relationship injection DISABLED.
    Used to isolate and prove the contribution of the relationship graph.
    """
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError("ChromaDB not found. Please run System Sync first.")

    from langchain_groq import ChatGroq
    from langchain_core.output_parsers import StrOutputParser

    vectorstore = get_vectorstore()
    retriever = _build_hybrid_retriever(vectorstore)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    ablation_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an E-Governance AI assistant. Answer the question based on the provided policy context.\n"
         "Treat all retrieved documents equally — do not apply any special logic for amendments.\n\n"
         "Context:\n{context}"),
        ("human", "{input}"),
    ])
    chain = ablation_prompt | llm | StrOutputParser()

    class AblationWrapper:
        def invoke(self, inputs):
            query = inputs["input"]
            t0 = time.time()
            docs = list(retriever.invoke(query))
            # NO relationship injection — this is the ablation condition
            context = "\n\n---\n\n".join(d.page_content for d in docs)
            ans = chain.invoke({"context": context, "input": query})
            latency_ms = round((time.time() - t0) * 1000)
            return {"answer": ans, "context": docs, "latency_ms": latency_ms}

    return AblationWrapper()
