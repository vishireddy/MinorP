import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key or api_key.startswith("gsk-your"):
    print("ERROR: Invalid GROQ_API_KEY found in .env")
    exit(1)

from src.retrieval_engine import get_vectorstore, create_relationship_aware_rag_chain
from src.ingestion import load_and_chunk_pdfs
from src.metadata_tagger import enrich_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_naive_rag_chain():
    """Creates a basic RAG chain that suffers from amendment blindness"""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant. Answer the question based on the context provided. Do not use outside knowledge.\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)
        
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def run_evaluation_suite(progress_callback=None):
    import time
    if not os.path.exists("data/chroma_db"):
        raise FileNotFoundError("Vector Database is empty. Please run System Sync first.")
        
    test_suite = [
        {"query": "According to the active legislation, what is the exact term of office for the Information Commissioner?", "amendment_truth": "central government"},
        {"query": "What is the minimum age required to be registered as a voter?", "amendment_truth": "18"},
        {"query": "How is Cyber Terrorism penalized under the IT Act?", "amendment_truth": "life"},
        {"query": "Does the Consumer Protection Act establish liability for e-commerce platforms?", "amendment_truth": "e-commerce"},
        {"query": "Is the term limit for an Information commissioner strictly defined as five years?", "amendment_truth": "central"}
    ]
    
    naive_success = 0
    aware_success = 0
    total = len(test_suite)
    
    naive_chain = create_naive_rag_chain()
    aware_chain = create_relationship_aware_rag_chain()
    
    results = []
    
    for i, test in enumerate(test_suite):
        if progress_callback:
            progress_callback(i / total, f"Evaluating Query {i+1}/{total}...")
        time.sleep(0.5) # Prevent API rate limiting
        
        # Test Naive RAG
        n_pass = False
        try:
            naive_ans = naive_chain.invoke(test["query"]).lower()
            if test["amendment_truth"].lower() in naive_ans:
                naive_success += 1
                n_pass = True
        except Exception:
            pass
            
        # Test Aware RAG
        a_pass = False
        try:
            aware_ans = aware_chain.invoke({"input": test["query"]})["answer"].lower()
            if test["amendment_truth"].lower() in aware_ans:
                aware_success += 1
                a_pass = True
        except Exception:
            pass
            
        results.append({
            "query": test["query"],
            "naive_pass": n_pass,
            "aware_pass": a_pass
        })
            
    if progress_callback:
        progress_callback(1.0, "Evaluation Complete!")
            
    naive_accuracy = (naive_success / total) * 100
    aware_accuracy = (aware_success / total) * 100
    improvement = aware_accuracy - naive_accuracy
    
    return {
        "metrics": {
            "naive_accuracy": naive_accuracy,
            "aware_accuracy": aware_accuracy,
            "improvement": improvement,
            "total_queries": total
        },
        "breakdown": results
    }

if __name__ == "__main__":
    res = run_evaluation_suite(lambda p, m: print(f"[{p*100:.0f}%] {m}"))
    print(res["metrics"])

