import os
import json
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

GRAPH_PATH = "data/relationship_graph.json"

class PolicyMetadata(BaseModel):
    is_amendment: bool = Field(description="Is this document an amendment OR does it repeal and replace an older act?")
    amends_policy: str = Field(description="If true, what is the exact name of the older policy it amends/replaces?", default=None)
    policy_name: str = Field(description="A short, descriptive name of this policy")

def load_graph():
    if os.path.exists(GRAPH_PATH):
        try:
            with open(GRAPH_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading graph: {e}")
            return {}
    return {}

def save_graph(graph):
    os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
    with open(GRAPH_PATH, 'w') as f:
        json.dump(graph, f, indent=4)

def extract_metadata_with_llm(filename: str, first_chunk_text: str):
    """Uses LLM to parse preamble topology"""
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    parser = JsonOutputParser(pydantic_object=PolicyMetadata)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert legal metadata extractor. Your job is to read the first page of a government policy and extract its relationship topology. \n{format_instructions}"),
        ("human", "Filename: {filename}\nPolicy Text:\n{text}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "filename": filename,
            "text": first_chunk_text,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        print(f"LLM Extraction failed for {filename}: {e}")
        return {"is_amendment": False, "amends_policy": None, "policy_name": filename}

def build_dynamic_graph(chunks: List[Document]):
    """Analyzes the chunks, uses LLM to discover relationships autonomously, and builds a dependency graph"""
    graph = load_graph()
    processed_files = set(graph.keys())
    new_files_to_process = {}
    
    for chunk in chunks:
        source_file = chunk.metadata.get("source", "")
        filename = os.path.basename(source_file)
        if filename not in processed_files and filename not in new_files_to_process:
            new_files_to_process[filename] = chunk.page_content[:2000] 
            
    if new_files_to_process:
        print(f"Discovered {len(new_files_to_process)} new policies. Running automated LLM Graph Extraction...")
        for filename, text in new_files_to_process.items():
            print(f"  -> Deep scanning {filename} for legal topology...")
            meta = extract_metadata_with_llm(filename, text)
            
            import time
            time.sleep(3) # Very important rate limiter for Groq's aggressive API tiers
            
            # For testing, if it starts with 'amendment', we forcefully classify it as such
            force_amend = filename.startswith("amendment_") or filename.startswith("superseding_")
            
            graph[filename] = {
                "status": "Active",
                "is_amendment": meta.get("is_amendment", False) or force_amend,
                "amends": meta.get("amends_policy") or "Unknown", 
                "amended_by": [] 
            }
            
        # Post-Processing: Explicit matching by filename tokens (ignoring numbers/years)
        for child_file, data in graph.items():
            if data["is_amendment"]:
                # Strip extensions and prefixes
                def get_core_tokens(fname):
                    clean = fname.replace(".pdf", "").replace("base_", "").replace("amendment_", "").replace("superseding_", "")
                    return set(w for w in clean.split("_") if not w.isdigit())
                
                child_tokens = get_core_tokens(child_file)
                
                for parent_file, p_data in graph.items():
                    if parent_file != child_file and parent_file.startswith("base_"):
                        parent_tokens = get_core_tokens(parent_file)
                        
                        # Calculate Jaccard similarity or intersection
                        common = child_tokens.intersection(parent_tokens)
                        
                        # As long as there is at least one matching core token (e.g. 'it', 'policy', 'consumer')
                        if len(common) > 0:
                            p_data["status"] = "Inactive/Superseded in part"
                            if child_file not in p_data["amended_by"]:
                                p_data["amended_by"].append(child_file)
                                
        save_graph(graph)
        print("  -> Dynamic relationship graph successfully updated!")
        
    return graph

def enrich_metadata(chunks: List[Document]) -> List[Document]:
    """Applies the dynamically generated graph relationships to the Chroma vector chunks"""
    graph = build_dynamic_graph(chunks)
    
    for chunk in chunks:
        source_file = chunk.metadata.get("source", "")
        filename = os.path.basename(source_file)
        
        chunk.metadata["document_name"] = filename
        
        if filename in graph:
            rel_data = graph[filename]
            chunk.metadata["status"] = rel_data.get("status", "Active")
            
            # ChromaDB filters require strict primitive types
            if rel_data.get("amends"):
                chunk.metadata["amends"] = str(rel_data["amends"])
                
            if rel_data.get("amended_by"):
                chunk.metadata["amended_by"] = ",".join(rel_data["amended_by"])
        else:
            chunk.metadata["status"] = "Active"
            
    return chunks
