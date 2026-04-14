import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

print("Checking API Key...")
if not api_key or api_key.startswith("gsk-your"):
    print("ERROR: Invalid GROQ_API_KEY found in .env")
    exit(1)
print("✅ API Key found!")

from src.ingestion import load_and_chunk_pdfs
from src.metadata_tagger import enrich_metadata
from src.retrieval_engine import get_vectorstore, create_relationship_aware_rag_chain

print("\n--- 1. Testing Indexing Pipeline ---")
try:
    chunks = load_and_chunk_pdfs()
    tagged_chunks = enrich_metadata(chunks)
    get_vectorstore(tagged_chunks)
    print("✅ Indexing successful!")
except Exception as e:
    print(f"❌ Indexing failed: {e}")
    exit(1)

print("\n--- 2. Testing AI Query (Relationship Check) ---")
try:
    rag_chain = create_relationship_aware_rag_chain()
    query = "What is the term of office for the Information Commissioner?"
    print(f"Query: '{query}'\n")
    
    response = rag_chain.invoke({"input": query})
    
    print("✅ AI Answer:")
    print("-" * 40)
    print(response["answer"])
    print("-" * 40)
    
    print("\n✅ Internal Sources Used by AI:")
    for doc in response.get("context", []):
        print(f" - {doc.metadata.get('document_name')} | Status: {doc.metadata.get('status')} | Content snippet: {doc.page_content[:30]}...")
        
except Exception as e:
    print(f"❌ AI Generation failed: {e}")
    exit(1)
