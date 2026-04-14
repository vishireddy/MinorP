import os
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_chunk_pdfs(data_dir: str = "data/raw", chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Loads all PDFs from the specified directory and splits them into chunks.
    Preserves page numbers and source file names in the document metadata.
    """
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Warning: No files found in {data_dir}. Please add PDF files.")
        return []

    print(f"Loading PDFs from {data_dir}...")
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages. Chunking...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\\n\\n", "\\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    
    # Add unique chunk IDs
    for idx, chunk in enumerate(chunks):
        basename = os.path.basename(chunk.metadata.get('source', 'unknown'))
        page = chunk.metadata.get('page', 0)
        chunk.metadata["chunk_id"] = f"{basename}-p{page}-c{idx}"
        
    return chunks

if __name__ == "__main__":
    # Test the ingestion function directly
    chunks = load_and_chunk_pdfs()
    if chunks:
        print(f"Sample chunk: {chunks[0].page_content[:100]}...")
        print(f"Sample metadata: {chunks[0].metadata}")
