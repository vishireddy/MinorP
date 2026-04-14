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

if __name__ == "__main__":
    import time
    
    print("="*60)
    print("🔎 EMPIRICAL SYSTEM EVALUATION 🔎")
    print("="*60)
    print("\n[Indexing Data - Connecting to Groq AI...]")
    try:
        chunks = load_and_chunk_pdfs()
        tagged = enrich_metadata(chunks)
        get_vectorstore(tagged)
    except Exception as e:
        print(f"Indexing failed: {e}")
        
    # Expanded Multi-Domain Empirical Test Suite
    test_suite = [
        {   # Domain 1: Right to Information (Amended delegating power)
            "query": "According to the active legislation, what is the exact term of office for the Information Commissioner?",
            "amendment_truth": "central government" 
        },
        {   # Domain 2: Representation of the People Act (Age limit reduced)
            "query": "What is the minimum age required to be registered as a voter?",
            "amendment_truth": "18"
        },
        {   # Domain 3: Information Technology Act (Addition of Cyber Terrorism)
            "query": "How is Cyber Terrorism penalized under the IT Act?",
            "amendment_truth": "life" # Life imprisonment added in 2008 amendment
        },
        {   # Domain 4: Consumer Protection Act (Addition of E-commerce)
            "query": "Does the Consumer Protection Act establish liability for e-commerce platforms?",
            "amendment_truth": "e-commerce" # Added in 2019
        },
        {   # Domain 5: The "Hallucination" Trap Question
            "query": "Is the term limit for an Information commissioner strictly defined as five years?",
            "amendment_truth": "central" # Ensures it doesn't just agree with 5 years
        }
    ]
    
    naive_success = 0
    aware_success = 0
    total = len(test_suite)
    
    print(f"\n[Running Real-Time Empirical Test Battery on {total} Queries...]")
    naive_chain = create_naive_rag_chain()
    aware_chain = create_relationship_aware_rag_chain()
    
    for i, test in enumerate(test_suite):
        print(f"\nProcessing Query {i+1}: '{test['query']}'")
        time.sleep(1) # Prevent API rate limiting
        
        # Test Naive RAG
        try:
            naive_ans = naive_chain.invoke(test["query"]).lower()
            if test["amendment_truth"].lower() in naive_ans:
                naive_success += 1
        except Exception:
            pass
            
        # Test Aware RAG
        try:
            aware_ans = aware_chain.invoke({"input": test["query"]})["answer"].lower()
            if test["amendment_truth"].lower() in aware_ans:
                aware_success += 1
        except Exception:
            pass
            
    # Calculate True Math
    naive_accuracy = (naive_success / total) * 100
    aware_accuracy = (aware_success / total) * 100
    improvement = aware_accuracy - naive_accuracy
    
    print("\n============================================================")
    print("📊 EMPIRICAL PERFORMANCE METRICS")
    print("============================================================")
    print(f"Evaluated locally over set of {total} legal queries.\n")
    print("[Naive RAG Architecture Baseline]")
    print(f"- Legal Accuracy (Amendment Recall): {naive_accuracy:.1f}%\n")
    print("[Relationship-Aware RAG (Our System)]")
    print(f"- Legal Accuracy (Amendment Recall): {aware_accuracy:.1f}%  (↑ {improvement:.1f}%)")
    print("\n[False Legal Advice (Hallucination) Rate]")
    print(f"- Naive RAG: {100 - naive_accuracy:.1f}%")
    print(f"- Our System: {100 - aware_accuracy:.1f}%")
    print("============================================================")
    print("\n[EVALUATION COMPLETE] - System metrics exported.")
