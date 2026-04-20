import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain_core.documents import Document

CHROMA_PATH = "data/chroma_db"

def get_vectorstore(chunks: List[Document] = None):
    """
    Initializes or loads the Chroma vector database.
    If chunks are provided, it adds them to the database.
    """
    # Using a free, lightweight open-source embedding model for fast local processing
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if chunks:
        print(f"Creating/Updating ChromaDB with {len(chunks)} chunks at {CHROMA_PATH}...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
    else:
        # Load existing
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        
    return vectorstore

def create_relationship_aware_rag_chain():
    """
    Creates the final RAG chain (Retrieval Augmented Generation).
    Uses GPT-3.5-turbo to generate answers, and explicitly instructs the LLM
    to pay attention to relationship metadata tags to prevent amendment blindness.
    """
    # Ensure ChromaDB exists before creating the chain
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError("ChromaDB not found. Please upload PDFs first.")

    vectorstore = get_vectorstore()
    # 1. Standard Vector Retriever (Semantic)
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 8  # Fetch Top 8 chunks for better recall across large document sets
        }
    )
    
    # 2. BM25 Lexical Retriever (Keyword)
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    
    # Extract raw data to initialize BM25 memory space
    raw_data = vectorstore.get()
    
    if raw_data and "documents" in raw_data and raw_data["documents"]:
        bm25_docs = []
        for doc_text, meta in zip(raw_data["documents"], raw_data["metadatas"]):
            bm25_docs.append(Document(page_content=doc_text, metadata=meta))
            
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = 8
        
        # 3. Create Custom Hybrid Ensemble (Reciprocal Rank Fusion)
        class HybridRRFRetriever:
            def __init__(self, bm25, vector, weight_bm25=0.3, weight_vec=0.7):
                self.bm25 = bm25
                self.vector = vector
                self.w_bm25 = weight_bm25
                self.w_vec = weight_vec

            def invoke(self, query):
                docs_bm25 = self.bm25.invoke(query)
                docs_vec = self.vector.invoke(query)
                
                rrf_scores = {}
                doc_map = {}
                k = 60 # RRF constant
                
                for rank, d in enumerate(docs_bm25):
                    score = self.w_bm25 * (1.0 / (rank + k))
                    doc_map[d.page_content] = d
                    rrf_scores[d.page_content] = rrf_scores.get(d.page_content, 0.0) + score
                    
                for rank, d in enumerate(docs_vec):
                    score = self.w_vec * (1.0 / (rank + k))
                    doc_map[d.page_content] = d
                    rrf_scores[d.page_content] = rrf_scores.get(d.page_content, 0.0) + score
                    
                # Sort by highest RRF score
                top_contents = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
                return [doc_map[c] for c in top_contents[:8]]
                
        retriever = HybridRRFRetriever(bm25_retriever, vector_retriever)
    else:
        retriever = vector_retriever
    
    # Requires GROQ_API_KEY in .env
    from langchain_groq import ChatGroq
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    # Prompt explicitly instructs the LLM to use our relationship tracking
    system_prompt = (
        "You are a helpful, citizen-facing E-Governance AI assistant.\n"
        "Use the provided policy context to answer the user's question simply and clearly.\n"
        "CRITICAL INSTRUCTION: Pay VERY CLOSE ATTENTION to the relationship status of the documents in the context.\n"
        "If you see a document tagged as 'Inactive/Superseded' and another document tagged as 'Active' or 'amends',\n"
        "you MUST base your final recommendation on the Active amendment to prevent 'amendment blindness'.\n"
        "Explain the policy updates to the user (e.g., 'Initially, the rule was X, but it was amended to Y').\n"
        "Always cite your sources, providing the exact Document Name.\n"
        "If the Context is completely empty or completely irrelevant to the question, gracefully apologize and state that the information is not present in the Active Policy Database.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    simple_chain = (
        prompt | llm | StrOutputParser()
    )
    
    # We create a simple wrapper class to mimic the output format of RetrievalChain
    class RAGWrapper:
        def invoke(self, inputs):
            query = inputs["input"]
            docs = retriever.invoke(query)
            
            # CRITICAL FIX: Langchain often returns immutable tuples. We MUST convert to a mutable list!
            docs = list(docs)
            
            # --- RELATIONSHIP-AWARE CONTEXT INJECTION FLAG ---
            import json, os
            graph_path = "data/relationship_graph.json"
            graph = {}  # Initialize empty graph — avoids NameError if file missing
            if os.path.exists(graph_path):
                try:
                    with open(graph_path, "r") as f:
                        graph = json.load(f)
                    
                    retrieved_files = set([os.path.basename(d.metadata.get("source", "")) for d in docs])
                    injected_docs = []
                    
                    for fname in list(retrieved_files):
                        if fname in graph and graph[fname].get("amended_by"):
                            for amendment_file in graph[fname]["amended_by"]:
                                if amendment_file not in retrieved_files:
                                    print(f"\n🔗 RELATIONSHIP DETECTED! Forcefully injecting '{amendment_file}' to resolve Semantic Gap...")
                                    try:
                                        # Semantic gap means similarity_search might fail to find it even with k=100
                                        # Absolute paths in Chroma vary, so we extract raw chunks matching the filename directly!
                                        all_data = vectorstore.get()
                                        extra_docs = []
                                        
                                        if all_data and "metadatas" in all_data:
                                            for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
                                                if amendment_file in meta.get("source", ""):
                                                    from langchain_core.documents import Document
                                                    extra_docs.append(Document(page_content=doc_text, metadata=meta))
                                                    if len(extra_docs) >= 3: # Pull top 3 chunks of the amendment
                                                        break
                                        
                                        if extra_docs:
                                            injected_docs.extend(extra_docs)
                                            retrieved_files.add(amendment_file)
                                        else:
                                            print(f"Failed to locate {amendment_file} in raw database.")
                                    except Exception as e:
                                        print("Injection failed:", e)
                                        
                    if injected_docs:
                        docs.extend(injected_docs)
                except Exception as e:
                    pass
            # ----------------------------------------------------
            
            # Format the documents to explicitly show the LLM which ones are superseded!
            formatted_chunks = []
            for d in docs:
                fname = os.path.basename(d.metadata.get("source", "Unknown"))
                # Default status to active if not in graph just in case
                status = graph.get(fname, {}).get("status", "Active") if 'graph' in locals() else "Unknown"
                
                chunk_str = f"📜 Document: {fname}\n"
                chunk_str += f"⚠️ RELATIONSHIP STATUS: {status}\n"
                
                # TELL THE LLM WHAT TO LOOK FOR IF SUPERSEDED!
                amended_by = graph.get(fname, {}).get("amended_by", []) if 'graph' in locals() else []
                if amended_by:
                    chunk_str += f"⚠️ SUPERSEDED BY: {', '.join(amended_by)}\n"
                    
                chunk_str += f"Text:\n{d.page_content}"
                formatted_chunks.append(chunk_str)
                
            ans = simple_chain.invoke({
                "context": "\n\n---\n\n".join(formatted_chunks),
                "input": query
            })
            return {"answer": ans, "context": docs}
            
    return RAGWrapper()
