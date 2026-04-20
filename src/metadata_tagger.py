import os
import json
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

GRAPH_PATH = "data/relationship_graph.json"

# Stopwords that must NOT count as meaningful shared tokens
STOPWORDS = {"act", "the", "of", "india", "and", "or", "for", "to", "a", "an", "in",
             "base", "amendment", "policy", "rules", "regulations", "code", "law", "bill"}

# Curated manual overrides: amendment file -> base file it amends
# This guarantees correct mapping regardless of filename conventions
MANUAL_OVERRIDES = {
    "amendment_bnss_2023.pdf":              "base_ccp1973.pdf",
    "amendment_bsa_2023.pdf":               "base_iea_1872.pdf",
    "amendment_it_act_2008.pdf":            "base_it_act_2000.pdf",
    "amendment_dpdp_2023.pdf":              "base_it_act_2000.pdf",
    "amendment_consumer_protection_2019.pdf": "base_consumer_protection_1986.pdf",
    "amendment_rpact_1989.pdf":             "base_rpact_1950.pdf",
    "amendment_rpact_1988.pdf":             "base_rpact_1950.pdf",
    "amendment_rpact_1951.pdf":             "base_rpact_1950.pdf",
    "amendment_rti_2019.pdf":               "base_rti_2005.pdf",
}


class PolicyMetadata(BaseModel):
    is_amendment: bool = Field(description="Is this document an amendment OR does it repeal and replace an older act?")
    amends_policy: str = Field(description="If is_amendment is true, what is the exact name of the older policy it amends/replaces? Otherwise return 'Unknown'.", default="Unknown")
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
        ("system", "You are an expert legal metadata extractor. Read the first page of a government policy and extract its relationship topology.\n{format_instructions}"),
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
        return {"is_amendment": False, "amends_policy": "Unknown", "policy_name": filename}


def get_meaningful_tokens(fname: str) -> set:
    """Extract meaningful tokens from filename, excluding stopwords and digits."""
    clean = fname.replace(".pdf", "").replace("base_", "").replace("amendment_", "").replace("superseding_", "")
    tokens = set(w for w in clean.split("_") if not w.isdigit() and w not in STOPWORDS and len(w) > 2)
    return tokens


def build_dynamic_graph(chunks: List[Document]):
    """
    Analyzes chunks, uses LLM to discover relationships, and builds a corrected dependency graph.
    Uses manual overrides + strict token matching (≥2 meaningful common tokens) to prevent false links.
    """
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
            time.sleep(2)  # Rate limiter for Groq API

            force_amend = filename.startswith("amendment_") or filename.startswith("superseding_")

            graph[filename] = {
                "status": "Active",
                "is_amendment": meta.get("is_amendment", False) or force_amend,
                "amends": meta.get("amends_policy") or "Unknown",
                "amended_by": []
            }

    # ── STEP 1: Apply manual curated overrides (highest priority, always correct) ──
    for amendment_file, base_file in MANUAL_OVERRIDES.items():
        if amendment_file in graph and base_file in graph:
            # Mark base as superseded
            if amendment_file not in graph[base_file]["amended_by"]:
                graph[base_file]["amended_by"].append(amendment_file)
                graph[base_file]["status"] = "Inactive/Superseded in part"
            # Ensure amendment amends field is set correctly
            if graph[amendment_file].get("amends") in [None, "Unknown"]:
                graph[amendment_file]["amends"] = base_file

    # ── STEP 2: Token-based fuzzy matching for new files not in manual overrides ──
    # Only run for new files, and require ≥ 2 MEANINGFUL shared tokens
    newly_added = set(new_files_to_process.keys()) if new_files_to_process else set()

    for child_file in newly_added:
        data = graph.get(child_file, {})
        if not data.get("is_amendment"):
            continue

        # Skip files already handled by manual overrides
        if child_file in MANUAL_OVERRIDES:
            continue

        child_tokens = get_meaningful_tokens(child_file)
        if not child_tokens:
            continue

        best_match = None
        best_score = 0

        for parent_file, p_data in graph.items():
            if parent_file == child_file or not parent_file.startswith("base_"):
                continue

            parent_tokens = get_meaningful_tokens(parent_file)
            common = child_tokens.intersection(parent_tokens)

            # CRITICAL: require at least 2 meaningful shared tokens
            if len(common) >= 2 and len(common) > best_score:
                best_score = len(common)
                best_match = parent_file

        if best_match and child_file not in graph[best_match]["amended_by"]:
            graph[best_match]["amended_by"].append(child_file)
            graph[best_match]["status"] = "Inactive/Superseded in part"
            print(f"  -> Token match: {child_file} amends {best_match} (shared: {best_score} tokens)")

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

            if rel_data.get("amends") and rel_data["amends"] != "Unknown":
                chunk.metadata["amends"] = str(rel_data["amends"])

            if rel_data.get("amended_by"):
                chunk.metadata["amended_by"] = ",".join(rel_data["amended_by"])
        else:
            chunk.metadata["status"] = "Active"

    return chunks
