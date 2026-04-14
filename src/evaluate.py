import os
import time
from dotenv import load_dotenv

load_dotenv()

from src.retrieval_engine import get_vectorstore, create_relationship_aware_rag_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================================
# 50-QUESTION BENCHMARK SUITE
# Grounded in the actual uploaded PDFs in data/raw/
# ============================================================
TEST_SUITE = [
    # --- GROUP 1: Electoral Law (RP Acts) ---
    {"query": "What is the minimum voting age in India as per the current law?",
     "truth_keywords": ["18"], "category": "Electoral Law", "tricky": True},
    {"query": "Are Electronic Voting Machines legally recognized under the Representation of the People Act?",
     "truth_keywords": ["electronic", "1988"], "category": "Electoral Law", "tricky": True},
    {"query": "Can a person convicted of a criminal offence contest elections in India?",
     "truth_keywords": ["disqualif"], "category": "Electoral Law", "tricky": False},
    {"query": "Under which section of the RP Act is corrupt practice at elections defined?",
     "truth_keywords": ["123"], "category": "Electoral Law", "tricky": False},
    {"query": "What authority is responsible for delimitation of constituencies in India?",
     "truth_keywords": ["delimitation", "election"], "category": "Electoral Law", "tricky": False},

    # --- GROUP 2: IT Act 2000 + IT Amendment Act 2008 ---
    {"query": "What is the penalty for cyber terrorism under the Information Technology Act?",
     "truth_keywords": ["life", "imprisonment"], "category": "IT & Cyber Law", "tricky": True},
    {"query": "What does Section 43A of the IT Act say about data protection for companies?",
     "truth_keywords": ["sensitive", "compensation"], "category": "IT & Cyber Law", "tricky": True},
    {"query": "Is sending offensive electronic messages a criminal offence under the IT Act?",
     "truth_keywords": ["offensive", "message"], "category": "IT & Cyber Law", "tricky": False},
    {"query": "What is a Digital Signature Certificate and who issues it under the IT Act?",
     "truth_keywords": ["certifying authority", "digital signature"], "category": "IT & Cyber Law", "tricky": False},
    {"query": "Under the IT Act, what is the role of CERT-In?",
     "truth_keywords": ["cyber", "incident"], "category": "IT & Cyber Law", "tricky": False},

    # --- GROUP 3: DPDP Act 2023 vs IT Act (Amendment Blindness) ---
    {"query": "Under current law, what is the maximum penalty for a data breach by a company in India?",
     "truth_keywords": ["250", "crore"], "category": "Data Privacy", "tricky": True},
    {"query": "Who is a Data Fiduciary under the DPDP Act 2023?",
     "truth_keywords": ["fiduciary", "personal data"], "category": "Data Privacy", "tricky": False},
    {"query": "Can the government access personal data without consent under the DPDP Act?",
     "truth_keywords": ["exempt", "security"], "category": "Data Privacy", "tricky": True},
    {"query": "What are the rights of a Data Principal under DPDP Act 2023?",
     "truth_keywords": ["correction", "erasure"], "category": "Data Privacy", "tricky": False},
    {"query": "What body adjudicates data protection violations under the DPDP Act 2023?",
     "truth_keywords": ["data protection board"], "category": "Data Privacy", "tricky": False},

    # --- GROUP 4: Consumer Protection 1986 vs 2019 ---
    {"query": "Where can a consumer file a complaint against an online seller under the current Consumer Protection Act?",
     "truth_keywords": ["residence", "complainant"], "category": "Consumer Law", "tricky": True},
    {"query": "What is the Central Consumer Protection Authority and what powers does it have?",
     "truth_keywords": ["ccpa", "misleading"], "category": "Consumer Law", "tricky": True},
    {"query": "What is the pecuniary limit for filing a case in the District Consumer Commission?",
     "truth_keywords": ["crore", "district"], "category": "Consumer Law", "tricky": True},
    {"query": "Is product liability recognized under the Consumer Protection Act?",
     "truth_keywords": ["product", "liability", "manufacturer"], "category": "Consumer Law", "tricky": False},
    {"query": "Can a consumer file a complaint for unfair trade practices under the 2019 Act?",
     "truth_keywords": ["unfair trade", "complaint"], "category": "Consumer Law", "tricky": False},

    # --- GROUP 5: Constitution of India ---
    {"query": "What does Article 21 of the Indian Constitution guarantee?",
     "truth_keywords": ["life", "personal liberty"], "category": "Constitutional Law", "tricky": False},
    {"query": "Which Article of the Constitution abolished untouchability?",
     "truth_keywords": ["17", "untouchability"], "category": "Constitutional Law", "tricky": False},
    {"query": "What is the significance of the 42nd Amendment to the Indian Constitution?",
     "truth_keywords": ["socialist", "secular", "fundamental duties"], "category": "Constitutional Law", "tricky": True},
    {"query": "Under which Article can the President declare a National Emergency?",
     "truth_keywords": ["352"], "category": "Constitutional Law", "tricky": False},
    {"query": "What is the doctrine of Basic Structure in Indian constitutional law?",
     "truth_keywords": ["basic structure", "parliament"], "category": "Constitutional Law", "tricky": True},

    # --- GROUP 6: RTI Act 2005 + RTI Amendment 2019 ---
    {"query": "What is the term of office of the Central Information Commissioner after the 2019 amendment?",
     "truth_keywords": ["central government", "term"], "category": "RTI", "tricky": True},
    {"query": "Can the RTI Act be used to obtain personal information about another citizen?",
     "truth_keywords": ["privacy", "personal", "exempt"], "category": "RTI", "tricky": False},
    {"query": "What exemptions are listed under Section 8 of the RTI Act?",
     "truth_keywords": ["sovereignty", "security", "cabinet"], "category": "RTI", "tricky": False},
    {"query": "Within how many days must a Public Information Officer respond to an RTI request?",
     "truth_keywords": ["30", "days"], "category": "RTI", "tricky": False},
    {"query": "Which authority hears second appeals under the RTI Act?",
     "truth_keywords": ["information commission", "second appeal"], "category": "RTI", "tricky": False},

    # --- GROUP 7: FEMA 1999 ---
    {"query": "What is the difference between a capital account and current account transaction under FEMA?",
     "truth_keywords": ["capital", "current", "reserve bank"], "category": "FEMA", "tricky": False},
    {"query": "What is the penalty for violating FEMA regulations?",
     "truth_keywords": ["three times", "penalty"], "category": "FEMA", "tricky": False},

    # --- GROUP 8: PMLA 2002 ---
    {"query": "What is money laundering as defined under PMLA 2002?",
     "truth_keywords": ["proceeds", "crime", "scheduled"], "category": "PMLA", "tricky": False},
    {"query": "Can property be attached before conviction under PMLA?",
     "truth_keywords": ["provisional", "attachment"], "category": "PMLA", "tricky": True},

    # --- GROUP 9: Competition Act 2002 ---
    {"query": "What is the role of the Competition Commission of India?",
     "truth_keywords": ["cci", "anti-competitive"], "category": "Competition Law", "tricky": False},
    {"query": "What threshold triggers mandatory merger notification to the CCI?",
     "truth_keywords": ["asset", "turnover", "combination"], "category": "Competition Law", "tricky": True},

    # --- GROUP 10: SEBI Act 1992 ---
    {"query": "What powers does SEBI have to protect investors?",
     "truth_keywords": ["investigate", "penalty", "securities"], "category": "Securities Law", "tricky": False},
    {"query": "Is insider trading prohibited under the SEBI Act?",
     "truth_keywords": ["insider", "unpublished", "price sensitive"], "category": "Securities Law", "tricky": False},

    # --- GROUP 11: RBI Act 1934 ---
    {"query": "What is the primary function of the Reserve Bank of India under the RBI Act?",
     "truth_keywords": ["monetary", "currency", "banking"], "category": "Banking Law", "tricky": False},
    {"query": "Who appoints the Governor of the Reserve Bank of India?",
     "truth_keywords": ["central government", "governor"], "category": "Banking Law", "tricky": False},

    # --- GROUP 12: SC/ST Act 1989 ---
    {"query": "What is the punishment for committing an atrocity against a Scheduled Caste member?",
     "truth_keywords": ["imprisonment", "atrocity"], "category": "SC/ST Act", "tricky": False},
    {"query": "Can an FIR under the SC/ST Atrocities Act be quashed by anticipatory bail?",
     "truth_keywords": ["anticipatory bail", "special court"], "category": "SC/ST Act", "tricky": True},

    # --- GROUP 13: RTE Act 2009 ---
    {"query": "What percentage of seats must private unaided schools reserve for economically weaker sections?",
     "truth_keywords": ["25", "economically weaker"], "category": "Education Law", "tricky": True},
    {"query": "Up to what age is free and compulsory education guaranteed under the RTE Act?",
     "truth_keywords": ["14", "compulsory"], "category": "Education Law", "tricky": False},

    # --- GROUP 14: Evidence Law (IEA 1872 + BSA 2023) ---
    {"query": "Under the new Bharatiya Sakshya Adhiniyam 2023, how is electronic evidence treated?",
     "truth_keywords": ["electronic", "document", "certificate"], "category": "Evidence Law", "tricky": True},
    {"query": "What is the dying declaration rule under Indian evidence law?",
     "truth_keywords": ["dying declaration", "death", "statement"], "category": "Evidence Law", "tricky": False},

    # --- GROUP 15: Forest Rights & Wildlife ---
    {"query": "What rights do tribals have over forest land under the Forest Rights Act 2006?",
     "truth_keywords": ["gram sabha", "forest", "habitat"], "category": "Environment Law", "tricky": False},
    {"query": "What is a Protected Area under the Wildlife Protection Act 1972?",
     "truth_keywords": ["sanctuary", "national park", "wildlife"], "category": "Environment Law", "tricky": False},

    # --- GROUP 16: Cross-Document Synthesis ---
    {"query": "Which Indian laws collectively protect a citizen's right to digital privacy and personal data?",
     "truth_keywords": ["article 21", "dpdp", "it act"], "category": "Cross-Document", "tricky": True},
    {"query": "If someone uses a forged electronic document to defraud a consumer online which laws apply?",
     "truth_keywords": ["it act", "consumer protection", "fraud"], "category": "Cross-Document", "tricky": True},
    {"query": "How do the Forest Rights Act and Wildlife Protection Act work together for conservation?",
     "truth_keywords": ["forest", "wildlife", "habitat", "gram sabha"], "category": "Cross-Document", "tricky": True},
]


def create_naive_chain():
    """Simulates a regular chatbot (ChatGPT/Gemini style) - same LLM, NO retrieval context."""
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledgeable AI assistant. Answer the question using your general training knowledge."),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


def create_naive_rag_chain():
    """Naive RAG - retrieves docs but ignores relationship metadata (amendment blindness)."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on the context only. Ignore which law version is older or newer.\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


def score_answer(answer: str, truth_keywords: list) -> bool:
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in truth_keywords)


def run_evaluation_suite(progress_callback=None):
    if not os.path.exists("data/chroma_db"):
        raise FileNotFoundError("Vector Database is empty. Please run System Sync first.")

    total = len(TEST_SUITE)
    naive_chain = create_naive_chain()
    aware_chain = create_relationship_aware_rag_chain()

    naive_success = aware_success = 0
    results = []
    category_scores = {}

    for i, test in enumerate(TEST_SUITE):
        if progress_callback:
            progress_callback(i / total, f"Q{i+1}/{total}: {test['query'][:55]}...")
        time.sleep(1)

        cat = test["category"]
        if cat not in category_scores:
            category_scores[cat] = {"naive": 0, "aware": 0, "total": 0}
        category_scores[cat]["total"] += 1

        n_pass = False
        try:
            naive_ans = naive_chain.invoke({"input": test["query"]})
            n_pass = score_answer(naive_ans, test["truth_keywords"])
            if n_pass:
                naive_success += 1
                category_scores[cat]["naive"] += 1
        except Exception:
            pass

        time.sleep(1)

        a_pass = False
        try:
            aware_resp = aware_chain.invoke({"input": test["query"]})
            aware_ans = aware_resp["answer"]
            a_pass = score_answer(aware_ans, test["truth_keywords"])
            if a_pass:
                aware_success += 1
                category_scores[cat]["aware"] += 1
        except Exception:
            pass

        results.append({
            "query": test["query"],
            "category": test["category"],
            "tricky": test["tricky"],
            "naive_pass": n_pass,
            "aware_pass": a_pass,
        })

    if progress_callback:
        progress_callback(1.0, "Evaluation Complete!")

    naive_accuracy = (naive_success / total) * 100
    aware_accuracy = (aware_success / total) * 100
    improvement = aware_accuracy - naive_accuracy

    tricky = [r for r in results if r["tricky"]]
    tricky_naive = sum(1 for r in tricky if r["naive_pass"])
    tricky_aware = sum(1 for r in tricky if r["aware_pass"])

    return {
        "metrics": {
            "total_queries": total,
            "naive_accuracy": naive_accuracy,
            "aware_accuracy": aware_accuracy,
            "improvement": improvement,
            "hallucination_rate": 100 - aware_accuracy,
            "tricky_total": len(tricky),
            "tricky_naive_accuracy": (tricky_naive / len(tricky)) * 100 if tricky else 0,
            "tricky_aware_accuracy": (tricky_aware / len(tricky)) * 100 if tricky else 0,
        },
        "breakdown": results,
        "category_scores": category_scores,
    }


if __name__ == "__main__":
    res = run_evaluation_suite(lambda p, m: print(f"[{p*100:.0f}%] {m}"))
    m = res["metrics"]
    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS ({m['total_queries']} Questions)")
    print(f"{'='*55}")
    print(f"  Naive Chatbot Accuracy   : {m['naive_accuracy']:.1f}%")
    print(f"  Aware RAG Accuracy       : {m['aware_accuracy']:.1f}%")
    print(f"  Improvement              : +{m['improvement']:.1f}%")
    print(f"  Hallucination Rate (RAG) : {m['hallucination_rate']:.1f}%")
    print(f"\n  Amendment-Trap Questions ({m['tricky_total']} Qs):")
    print(f"  Naive Chatbot            : {m['tricky_naive_accuracy']:.1f}%")
    print(f"  Aware RAG                : {m['tricky_aware_accuracy']:.1f}%")
    print(f"\n  Category Breakdown:")
    for cat, s in res["category_scores"].items():
        n_pct = (s['naive'] / s['total']) * 100
        a_pct = (s['aware'] / s['total']) * 100
        print(f"  {cat:<22}: Naive {n_pct:.0f}%  |  Aware {a_pct:.0f}%  ({s['total']} Qs)")
