import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

from src.retrieval_engine import get_vectorstore, create_relationship_aware_rag_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================================
# LLM-AS-A-JUDGE EVALUATION SUITE
# ─────────────────────────────────────────────────────────────
# Each question has a human-written REFERENCE ANSWER derived
# directly from the actual uploaded PDFs (Indian law texts).
#
# A separate "Judge LLM" reads:
#   - The question
#   - The reference answer (ground truth)
#   - The model's actual answer
# ...and scores it 0–10 based on legal accuracy and completeness.
# Score ≥ 6 = PASS. This eliminates all keyword-matching limits.
# ============================================================

TEST_SUITE = [
    # ── GROUP 1: Electoral Law ──────────────────────────────────
    {
        "query": "What is the minimum voting age to be registered as a voter in India?",
        "reference": "Under the Representation of the People Act 1950, as amended, a person must be at least 18 years of age to be registered as a voter in India.",
        "category": "Electoral Law", "tricky": True
    },
    {
        "query": "Are Electronic Voting Machines legally recognized under the Representation of the People Act?",
        "reference": "Yes. The Representation of the People (Amendment) Act 1988 inserted Section 61A into the RP Act 1951, which expressly recognises voting machines (Electronic Voting Machines) as a valid method of recording votes.",
        "category": "Electoral Law", "tricky": True
    },
    {
        "query": "What constitutes a corrupt practice at elections under the Representation of People Act?",
        "reference": "Section 123 of the Representation of the People Act 1951 defines corrupt practices, including bribery, undue influence, misrepresentation of a candidate's identity, publication of false statements, incitement to violence, and booth capturing.",
        "category": "Electoral Law", "tricky": False
    },
    {
        "query": "What is the process for disqualification of a candidate under election law?",
        "reference": "Under Sections 8–11 of the Representation of the People Act 1951, a person convicted of certain criminal offences can be disqualified from contesting elections. The disqualification period depends on the sentence. A person convicted with imprisonment of two years or more is disqualified for six years after release.",
        "category": "Electoral Law", "tricky": False
    },
    {
        "query": "What is the role of the Election Commission under the Representation of the People Act?",
        "reference": "The Election Commission of India, under Article 324 of the Constitution and the Representation of the People Acts, has superintendence, direction, and control over the preparation of electoral rolls and the conduct of all elections to Parliament and State Legislatures.",
        "category": "Electoral Law", "tricky": False
    },

    # ── GROUP 2: IT Act 2000 + Amendment 2008 ──────────────────
    {
        "query": "What is the penalty for cyber terrorism under the Information Technology Act?",
        "reference": "Section 66F of the Information Technology Act, inserted by the 2008 Amendment, prescribes imprisonment for life as the maximum penalty for cyber terrorism — acts that threaten national security, sovereignty, or public order through computer networks.",
        "category": "IT & Cyber Law", "tricky": True
    },
    {
        "query": "What protection does Section 43A of the IT Act provide for sensitive personal data?",
        "reference": "Section 43A, inserted by the IT Amendment Act 2008, requires body corporates handling sensitive personal data to implement and maintain reasonable security practices. Failure to do so, resulting in wrongful loss or gain, makes them liable to pay compensation to the affected person.",
        "category": "IT & Cyber Law", "tricky": True
    },
    {
        "query": "What does the IT Act say about electronic signatures?",
        "reference": "The IT Act recognises electronic signatures as legally valid equivalents of handwritten signatures for authenticating electronic records. A Certifying Authority issues Digital Signature Certificates. Section 5 grants electronic signatures the same legal standing as physical signatures.",
        "category": "IT & Cyber Law", "tricky": False
    },
    {
        "query": "What is the role of CERT-In under the Information Technology Act?",
        "reference": "CERT-In (Indian Computer Emergency Response Team) is established under Section 70B of the IT Act (inserted by the 2008 Amendment). It functions as the national nodal agency for responding to cyber security incidents, issuing guidelines, and coordinating cyber incident response.",
        "category": "IT & Cyber Law", "tricky": False
    },
    {
        "query": "What constitutes hacking under the IT Act?",
        "reference": "Section 66 of the IT Act (as amended in 2008) defines computer-related offences including dishonestly or fraudulently doing any act that causes wrongful loss, damage to data, or disruption to a computer system. This covers unauthorised access, data theft, and system damage.",
        "category": "IT & Cyber Law", "tricky": False
    },

    # ── GROUP 3: DPDP Act 2023 ─────────────────────────────────
    {
        "query": "What is the maximum penalty under the DPDP Act 2023 for a data breach?",
        "reference": "The Digital Personal Data Protection Act 2023 prescribes a maximum penalty of Rs 250 crore (two hundred and fifty crore rupees) for a data fiduciary that fails to implement adequate security measures to prevent a personal data breach. The Data Protection Board adjudicates these penalties.",
        "category": "Data Privacy", "tricky": True
    },
    {
        "query": "Who is a Data Fiduciary under the DPDP Act 2023?",
        "reference": "Under the DPDP Act 2023, a Data Fiduciary is any person, including the State, a company, or an individual, who alone or jointly with others determines the purpose and means of processing of personal data.",
        "category": "Data Privacy", "tricky": False
    },
    {
        "query": "What government exemptions exist under the DPDP Act for processing personal data?",
        "reference": "The DPDP Act 2023 exempts the Government from certain provisions when processing data for purposes of national security, sovereignty, public order, prevention of offences, and legal proceedings. The Central Government may also exempt certain Government entities by notification.",
        "category": "Data Privacy", "tricky": True
    },
    {
        "query": "What are the rights of a person whose data is processed under the DPDP Act 2023?",
        "reference": "The DPDP Act 2023 grants Data Principals rights including: (1) right to access information about their data, (2) right to correction and erasure of inaccurate data, (3) right to grievance redressal, (4) right to nominate a representative for data decisions in case of death or incapacity.",
        "category": "Data Privacy", "tricky": False
    },
    {
        "query": "What body is empowered to adjudicate complaints under the DPDP Act 2023?",
        "reference": "The Data Protection Board of India, established under the DPDP Act 2023, is empowered to adjudicate complaints, impose penalties on Data Fiduciaries, and direct remedies for Data Principals whose rights are violated.",
        "category": "Data Privacy", "tricky": False
    },

    # ── GROUP 4: Consumer Protection 1986 + 2019 Amendment ─────
    {
        "query": "Where can a consumer file a complaint against an online seller under current law?",
        "reference": "Under the Consumer Protection Act 2019, a consumer can file a complaint at the District Commission where the complainant resides or personally works for gain — not just where the seller is located. This is a major change from the 1986 Act introduced specifically to protect e-commerce consumers.",
        "category": "Consumer Law", "tricky": True
    },
    {
        "query": "What is the Central Consumer Protection Authority and what can it do?",
        "reference": "The Central Consumer Protection Authority (CCPA) was established under the Consumer Protection Act 2019. It can investigate unfair trade practices, order recall of unsafe goods, cancel misleading advertisements, impose penalties on advertisers, and file complaints in Consumer Commissions on behalf of consumers.",
        "category": "Consumer Law", "tricky": True
    },
    {
        "query": "What is the jurisdiction limit for the District Consumer Commission?",
        "reference": "Under the Consumer Protection Act 2019, the District Consumer Disputes Redressal Commission has pecuniary jurisdiction to hear complaints where the value of goods, services, or compensation claimed does not exceed one crore rupees.",
        "category": "Consumer Law", "tricky": True
    },
    {
        "query": "What is product liability under the Consumer Protection Act?",
        "reference": "Product liability under Chapter VI of the Consumer Protection Act 2019 makes manufacturers, product service providers, and product sellers liable to compensate consumers for harm caused by a defective product. This includes manufacturing defects, design defects, or inadequate instructions or warnings.",
        "category": "Consumer Law", "tricky": False
    },
    {
        "query": "What is an unfair trade practice under the Consumer Protection Act?",
        "reference": "An unfair trade practice under the Consumer Protection Act includes false representations about the quality, standard, or quantity of goods; misleading advertisements; offering gifts or prizes with no intention to give them; and other deceptive practices that induce consumers into unfair transactions.",
        "category": "Consumer Law", "tricky": False
    },

    # ── GROUP 5: Constitution of India ─────────────────────────
    {
        "query": "What fundamental right does Article 21 of the Indian Constitution protect?",
        "reference": "Article 21 of the Indian Constitution states that no person shall be deprived of his life or personal liberty except according to procedure established by law. The Supreme Court has broadly interpreted this to include the right to privacy, right to livelihood, right to education, and other inherent rights.",
        "category": "Constitutional Law", "tricky": False
    },
    {
        "query": "Which article of the Indian Constitution abolishes untouchability?",
        "reference": "Article 17 of the Indian Constitution abolishes untouchability in any form and makes its practice a punishable offence under law. The enforcement of any disability arising out of untouchability is prohibited.",
        "category": "Constitutional Law", "tricky": False
    },
    {
        "query": "Under which article can the President of India declare a National Emergency?",
        "reference": "Article 352 of the Indian Constitution empowers the President to proclaim a National Emergency if the security of India or any part is threatened by war, external aggression, or armed rebellion. The 44th Amendment replaced 'internal disturbance' with 'armed rebellion'.",
        "category": "Constitutional Law", "tricky": False
    },
    {
        "query": "What are Directive Principles of State Policy in the Indian Constitution?",
        "reference": "Directive Principles of State Policy (Part IV, Articles 36–51) are guidelines for the State to follow while formulating policies and laws. Unlike Fundamental Rights they are not justiceable — they cannot be directly enforced by courts — but are fundamental to governance and the State must apply them in making laws.",
        "category": "Constitutional Law", "tricky": False
    },
    {
        "query": "What does Article 32 of the Indian Constitution provide?",
        "reference": "Article 32 gives every citizen the right to approach the Supreme Court directly for enforcement of Fundamental Rights. The Supreme Court has the power to issue writs including habeas corpus, mandamus, prohibition, quo warranto, and certiorari. Dr Ambedkar called it the 'heart and soul' of the Constitution.",
        "category": "Constitutional Law", "tricky": False
    },

    # ── GROUP 6: RTI Act 2005 + Amendment 2019 ─────────────────
    {
        "query": "What is the term of office of the Central Information Commissioner after the 2019 amendment?",
        "reference": "The Right to Information (Amendment) Act 2019 removed the fixed five-year term for the Chief Information Commissioner and Information Commissioners. Now their term and conditions of service — including salary, allowances, and tenure — are determined by the Central Government.",
        "category": "RTI", "tricky": True
    },
    {
        "query": "What categories of information are exempt under Section 8 of the RTI Act?",
        "reference": "Section 8 of the RTI Act 2005 exempts from disclosure: information affecting sovereignty and integrity, information received from foreign governments, information prejudicial to national security, cabinet papers, information that would endanger life, trade secrets, and information held in fiduciary capacity.",
        "category": "RTI", "tricky": False
    },
    {
        "query": "Within how many days must a Public Information Officer respond to an RTI application?",
        "reference": "Under Section 7(1) of the RTI Act 2005, a Public Information Officer must provide the requested information within thirty days of receipt of the application. If the information concerns the life or liberty of a person, it must be provided within 48 hours.",
        "category": "RTI", "tricky": False
    },
    {
        "query": "What is the role of the First Appellate Authority under the RTI Act?",
        "reference": "Under Section 19 of the RTI Act, if an applicant receives no response or is unsatisfied, they can file a first appeal to the First Appellate Authority — an officer senior to the Public Information Officer — within 30 days. The FAA must decide within 30 days (extendable to 45).",
        "category": "RTI", "tricky": False
    },
    {
        "query": "What penalty can the Information Commission impose under the RTI Act?",
        "reference": "Under Section 20 of the RTI Act, the Information Commission can impose a penalty on the Public Information Officer of Rs 250 per day, subject to a maximum of Rs 25,000, for refusing to receive an application, delaying information, misrepresenting facts, or destroying information.",
        "category": "RTI", "tricky": False
    },

    # ── GROUP 7: FEMA 1999 ─────────────────────────────────────
    {
        "query": "What is the difference between capital account and current account transactions under FEMA?",
        "reference": "Under FEMA 1999, current account transactions are those that do not alter the assets or liabilities abroad — such as trade payments, remittances, and travel. Capital account transactions alter overseas assets or liabilities — such as investments, loans, and property acquisition. Current account transactions are generally free; capital account transactions require RBI permission.",
        "category": "FEMA", "tricky": False
    },
    {
        "query": "What penalties are prescribed for FEMA violations?",
        "reference": "Under Section 13 of FEMA 1999, a person who contravenes any provision can be penalised up to thrice the sum involved in the contravention, or up to two lakh rupees where the amount is not quantifiable. Continued violations attract further daily penalties. Adjudication is by an Adjudicating Authority appointed by the Central Government.",
        "category": "FEMA", "tricky": False
    },

    # ── GROUP 8: PMLA 2002 ─────────────────────────────────────
    {
        "query": "What is the legal definition of money laundering under PMLA 2002?",
        "reference": "Section 3 of the Prevention of Money Laundering Act 2002 defines money laundering as directly or indirectly attempting to indulge in, or knowingly assisting in, any process or activity connected with proceeds of crime — including concealment, possession, acquisition, or use of proceeds derived from a scheduled offence — and projecting them as untainted property.",
        "category": "PMLA", "tricky": False
    },
    {
        "query": "Can the Enforcement Directorate attach property before conviction under PMLA?",
        "reference": "Yes. Section 5 of PMLA 2002 empowers the Director or an authorised officer to provisionally attach property believed to be proceeds of crime for up to 180 days, before any conviction. The provisional attachment must be confirmed by the Adjudicating Authority and cannot be lifted merely because criminal proceedings are pending.",
        "category": "PMLA", "tricky": True
    },

    # ── GROUP 9: Competition Act 2002 ──────────────────────────
    {
        "query": "What is the Competition Commission of India's mandate under the Competition Act?",
        "reference": "The Competition Commission of India (CCI) is established under the Competition Act 2002 to prevent practices having adverse effects on competition, to promote and sustain competition, to protect consumer interests, and to ensure freedom of trade. It has powers to investigate anti-competitive agreements, abuse of dominant position, and regulate combinations.",
        "category": "Competition Law", "tricky": False
    },
    {
        "query": "What triggers a mandatory notification to the CCI for a merger?",
        "reference": "Under Section 6 of the Competition Act 2002, a combination — through merger, acquisition, or amalgamation — that crosses prescribed asset or turnover thresholds must be mandatorily notified to the CCI before completion. The CCI reviews whether the combination is likely to cause an appreciable adverse effect on competition.",
        "category": "Competition Law", "tricky": True
    },

    # ── GROUP 10: SEBI Act 1992 ────────────────────────────────
    {
        "query": "What powers does SEBI have to regulate the securities market?",
        "reference": "SEBI (Securities and Exchange Board of India) under the SEBI Act 1992 has powers to register and regulate stock brokers, sub-brokers, and other intermediaries; prohibit insider trading; investigate fraudulent practices; conduct audits; impose penalties; and issue regulations for orderly functioning of the securities market.",
        "category": "Securities Law", "tricky": False
    },
    {
        "query": "How does the SEBI Act define insider trading?",
        "reference": "Insider trading under SEBI regulations involves trading in securities of a company by a person who has access to unpublished price-sensitive information (UPSI) about the company. Such persons — insiders — are prohibited from trading, communicating, or counselling others to trade on the basis of UPSI.",
        "category": "Securities Law", "tricky": False
    },

    # ── GROUP 11: RBI Act 1934 ─────────────────────────────────
    {
        "query": "What are the primary functions of the Reserve Bank of India?",
        "reference": "The Reserve Bank of India Act 1934 establishes the RBI as the central bank. Its primary functions include: issuing currency notes, acting as banker to the Central and State Governments, regulating credit and monetary policy, managing foreign exchange reserves, and supervising the banking system.",
        "category": "Banking Law", "tricky": False
    },
    {
        "query": "Who has the power to appoint the Governor of the Reserve Bank of India?",
        "reference": "Under Section 8 of the RBI Act 1934, the Central Government appoints the Governor and Deputy Governors of the Reserve Bank of India.",
        "category": "Banking Law", "tricky": False
    },

    # ── GROUP 12: SC/ST Act 1989 ───────────────────────────────
    {
        "query": "What is the punishment for committing an atrocity against a Scheduled Caste member?",
        "reference": "The Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act 1989 prescribes imprisonment of not less than six months, extendable up to five years with fine, for most atrocity offences. For aggravated offences — such as murder or grievous hurt — the punishment can extend to life imprisonment.",
        "category": "SC/ST Act", "tricky": False
    },
    {
        "query": "Are special courts required to hear cases under the SC/ST Atrocities Act?",
        "reference": "Yes. The SC/ST (Prevention of Atrocities) Act 1989 mandates the establishment of Special Courts and Exclusive Special Courts for the speedy trial of atrocity offences. These courts have the powers of a Court of Sessions and must take up cases without a separate committal proceeding.",
        "category": "SC/ST Act", "tricky": True
    },

    # ── GROUP 13: RTE Act 2009 ─────────────────────────────────
    {
        "query": "What percentage of seats must private unaided schools reserve for weaker sections under RTE?",
        "reference": "Section 12(1)(c) of the Right to Education Act 2009 requires every private unaided school to admit at least 25 percent of its intake at the entry-level class from children belonging to weaker sections and disadvantaged groups in the neighbourhood, and provide free and compulsory education to them until completion of elementary education.",
        "category": "Education Law", "tricky": True
    },
    {
        "query": "Up to what age is free and compulsory education guaranteed under the RTE Act?",
        "reference": "The Right of Children to Free and Compulsory Education Act 2009 guarantees free and compulsory education to all children between the ages of six and fourteen years (i.e., Class 1 to Class 8) as a Fundamental Right under Article 21A of the Indian Constitution.",
        "category": "Education Law", "tricky": False
    },

    # ── GROUP 14: Evidence Law (IEA 1872 + BSA 2023) ───────────
    {
        "query": "How does the Bharatiya Sakshya Adhiniyam 2023 treat electronic records as evidence?",
        "reference": "The Bharatiya Sakshya Adhiniyam 2023 (which replaces the Indian Evidence Act 1872) treats electronic records as documentary evidence. Section 63 provides that electronic records are admissible as evidence. A certificate from the person responsible for the device, or from an expert, is required to authenticate electronic records, replacing the earlier Section 65B IEA certificate requirement.",
        "category": "Evidence Law", "tricky": True
    },
    {
        "query": "What is a dying declaration and when is it admissible under Indian evidence law?",
        "reference": "A dying declaration is a statement made by a person about the cause of their death or circumstances of the transaction that resulted in their death. Under Section 32(1) of the Indian Evidence Act 1872 (and its equivalent in the Bharatiya Sakshya Adhiniyam 2023), such a statement is admissible as evidence even though the person is not alive to be cross-examined, because the law presumes a dying person has no motive to lie.",
        "category": "Evidence Law", "tricky": False
    },

    # ── GROUP 15: Forest Rights & Wildlife ─────────────────────
    {
        "query": "What community rights does the Forest Rights Act 2006 grant to tribal communities?",
        "reference": "The Scheduled Tribes and Other Traditional Forest Dwellers (Recognition of Forest Rights) Act 2006 grants tribals and forest-dwelling communities rights to: (1) hold and live in forest land under individual or community tenure, (2) access, use, and manage community forest resources, (3) protect habitat and biodiversity, (4) convert forest villages to revenue villages, and (5) participate in Gram Sabha decisions on forest governance.",
        "category": "Environment Law", "tricky": False
    },
    {
        "query": "What constitutes a Protected Area under the Wildlife Protection Act 1972?",
        "reference": "Under the Wildlife Protection Act 1972, a Protected Area is any area declared by the Central or State Government as a National Park, Wildlife Sanctuary, Conservation Reserve, or Community Reserve. National Parks have the strictest protection — no human activity is permitted. Sanctuaries allow some regulated human activity. Both forms prohibit hunting within their boundaries.",
        "category": "Environment Law", "tricky": False
    },

    # ── GROUP 16: Cross-Document Synthesis ─────────────────────
    {
        "query": "Which Indian laws together protect a citizen's right to digital privacy?",
        "reference": "Digital privacy in India is protected by a combination of: (1) Article 21 of the Constitution (right to privacy as part of right to life, per Puttaswamy judgment), (2) the Digital Personal Data Protection Act 2023 (data fiduciary obligations, consent requirements), and (3) the IT Act 2000 with its 2008 Amendment (Section 43A on sensitive data, Section 72A on breach of confidentiality).",
        "category": "Cross-Document", "tricky": True
    },
    {
        "query": "If someone fraudulently uses an electronic document to cheat an online consumer, which laws apply?",
        "reference": "Multiple laws apply: (1) IT Act 2000 — Section 66D (cheating by personation using computer resource), Section 43 (data tampering), (2) Consumer Protection Act 2019 — unfair trade practice, product/service deficiency, CCPA can take suo motu action, (3) IPC/BNS — forgery and cheating provisions. The victim can also approach the District Consumer Commission for compensation.",
        "category": "Cross-Document", "tricky": True
    },
    {
        "query": "How do the Forest Rights Act and Wildlife Protection Act interact for forest conservation?",
        "reference": "The Forest Rights Act 2006 and Wildlife Protection Act 1972 interact by balancing community rights with conservation. While the WPA protects biodiversity through Protected Areas, the FRA recognises tribal rights to live in and manage forest habitat including inside sanctuaries. The Gram Sabha under FRA can conserve forests and wildlife as a community right, complementing the WPA's state-led protection. The FRA also gives tribals the right to protect their community forest resources from encroachment or damage.",
        "category": "Cross-Document", "tricky": True
    },
]


# ──────────────────────────────────────────────────────────────
# LLM-AS-A-JUDGE SCORER
# ──────────────────────────────────────────────────────────────
def create_judge_chain():
    """
    A separate powerful LLM that acts as an impartial examiner.
    It reads the reference answer and the model's answer, then
    scores factual accuracy and legal completeness on a 0–10 scale.
    Score >= 6 = PASS.
    """
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Indian legal examiner evaluating AI answers.
Given a question, a reference answer (ground truth), and a model's answer, score the model's answer from 0 to 10.

Scoring criteria:
- 9-10: Factually correct, complete, cites correct legal provisions
- 7-8: Mostly correct with minor omissions or slight imprecision
- 5-6: Partially correct — gets the main point but misses important details
- 3-4: Contains some relevant information but has significant errors
- 0-2: Wrong, irrelevant, or states it cannot answer

IMPORTANT: 
- Do NOT penalise for different wording or phrasing — only penalise for factual errors
- If the model says it cannot find information, score it 0-1
- Award full marks if the core legal fact is correct even if the answer is brief

Respond ONLY with a JSON object in this exact format:
{{"score": <integer 0-10>, "reason": "<one sentence explaining the score>"}}"""),
        ("human", """Question: {query}

Reference Answer: {reference}

Model's Answer: {model_answer}

Score:""")
    ])
    return judge_prompt | llm | StrOutputParser()


def judge_score(judge_chain, query: str, reference: str, model_answer: str, retries: int = 2) -> tuple[int, str]:
    """Returns (score 0-10, reason string)"""
    for attempt in range(retries + 1):
        try:
            raw = judge_chain.invoke({
                "query": query,
                "reference": reference,
                "model_answer": model_answer
            })
            # Parse JSON from judge output
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                score = int(data.get("score", 0))
                reason = data.get("reason", "")
                return max(0, min(10, score)), reason
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
    return 0, "Judge failed to evaluate"


# ──────────────────────────────────────────────────────────────
# MODEL CHAINS
# ──────────────────────────────────────────────────────────────
def create_naive_llm_chain():
    """Google Gemma2-9b via Groq — no documents, pure training knowledge."""
    llm = ChatGroq(model_name="gemma2-9b-it", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledgeable AI assistant. Answer the legal question using your training knowledge. Be specific and cite relevant legal provisions if you know them."),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


def create_naive_rag_chain():
    """Mistral Mixtral-8x7b via Groq — retrieves docs but ignores amendment relationships."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the provided context. Treat all documents equally.\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


# ──────────────────────────────────────────────────────────────
# MAIN EVALUATION RUNNER
# ──────────────────────────────────────────────────────────────
def run_evaluation_suite(progress_callback=None):
    if not os.path.exists("data/chroma_db"):
        raise FileNotFoundError("Vector Database is empty. Please run System Sync first.")

    total = len(TEST_SUITE)
    judge_chain = create_judge_chain()
    naive_llm_chain = create_naive_llm_chain()
    naive_rag_chain = create_naive_rag_chain()
    aware_chain = create_relationship_aware_rag_chain()

    naive_llm_total = naive_rag_total = aware_total = 0
    results = []
    category_scores = {}

    for i, test in enumerate(TEST_SUITE):
        if progress_callback:
            progress_callback(i / total, f"Q{i+1}/{total}: {test['query'][:55]}...")

        cat = test["category"]
        if cat not in category_scores:
            category_scores[cat] = {
                "naive_llm": 0.0, "naive_rag": 0.0, "aware": 0.0,
                "naive_llm_pass": 0, "naive_rag_pass": 0, "aware_pass": 0,
                "total": 0
            }
        category_scores[cat]["total"] += 1

        # ── 1. Naive LLM (Google Gemma2-9b, no docs) ──
        nl_score, nl_reason, nl_ans = 0, "Error", ""
        try:
            nl_ans = naive_llm_chain.invoke({"input": test["query"]})
            time.sleep(1)
            nl_score, nl_reason = judge_score(judge_chain, test["query"], test["reference"], nl_ans)
            naive_llm_total += nl_score
            category_scores[cat]["naive_llm"] += nl_score
            if nl_score >= 6:
                category_scores[cat]["naive_llm_pass"] += 1
        except Exception as e:
            nl_reason = str(e)
        time.sleep(1)

        # ── 2. Naive RAG (Mixtral, no amendment awareness) ──
        nr_score, nr_reason, nr_ans = 0, "Error", ""
        try:
            nr_ans = naive_rag_chain.invoke({"input": test["query"]})
            time.sleep(1)
            nr_score, nr_reason = judge_score(judge_chain, test["query"], test["reference"], nr_ans)
            naive_rag_total += nr_score
            category_scores[cat]["naive_rag"] += nr_score
            if nr_score >= 6:
                category_scores[cat]["naive_rag_pass"] += 1
        except Exception as e:
            nr_reason = str(e)
        time.sleep(1)

        # ── 3. Relationship-Aware RAG (LLaMA3, full system) ──
        aw_score, aw_reason, aw_ans = 0, "Error", ""
        try:
            aw_resp = aware_chain.invoke({"input": test["query"]})
            aw_ans = aw_resp["answer"]
            time.sleep(1)
            aw_score, aw_reason = judge_score(judge_chain, test["query"], test["reference"], aw_ans)
            aware_total += aw_score
            category_scores[cat]["aware"] += aw_score
            if aw_score >= 6:
                category_scores[cat]["aware_pass"] += 1
        except Exception as e:
            aw_reason = str(e)
        time.sleep(1)

        results.append({
            "query": test["query"],
            "reference": test["reference"],
            "category": cat,
            "tricky": test["tricky"],
            "naive_llm_score": nl_score,
            "naive_llm_reason": nl_reason,
            "naive_rag_score": nr_score,
            "naive_rag_reason": nr_reason,
            "aware_score": aw_score,
            "aware_reason": aw_reason,
            # Pass = score >= 6
            "naive_llm_pass": nl_score >= 6,
            "naive_rag_pass": nr_score >= 6,
            "aware_pass": aw_score >= 6,
        })

    if progress_callback:
        progress_callback(1.0, "Evaluation Complete!")

    # Accuracy = percentage of questions scoring >= 6/10
    nl_pass = sum(1 for r in results if r["naive_llm_pass"])
    nr_pass = sum(1 for r in results if r["naive_rag_pass"])
    aw_pass = sum(1 for r in results if r["aware_pass"])

    nl_acc = nl_pass / total * 100
    nr_acc = nr_pass / total * 100
    aw_acc = aw_pass / total * 100

    # Average judge scores (out of 10)
    nl_avg = naive_llm_total / total
    nr_avg = naive_rag_total / total
    aw_avg = aware_total / total

    tricky = [r for r in results if r["tricky"]]
    nt = len(tricky)

    return {
        "metrics": {
            "total_queries": total,
            # Pass rates (score >= 6)
            "naive_llm_accuracy": nl_acc,
            "naive_rag_accuracy": nr_acc,
            "aware_accuracy": aw_acc,
            "rag_improvement_over_llm": aw_acc - nl_acc,
            "rag_improvement_over_naive_rag": aw_acc - nr_acc,
            "hallucination_rate": 100 - aw_acc,
            # Average judge scores
            "naive_llm_avg_score": nl_avg,
            "naive_rag_avg_score": nr_avg,
            "aware_avg_score": aw_avg,
            # Tricky / amendment-trap
            "tricky_total": nt,
            "tricky_naive_llm_accuracy": sum(1 for r in tricky if r["naive_llm_pass"]) / nt * 100 if nt else 0,
            "tricky_naive_rag_accuracy": sum(1 for r in tricky if r["naive_rag_pass"]) / nt * 100 if nt else 0,
            "tricky_aware_accuracy": sum(1 for r in tricky if r["aware_pass"]) / nt * 100 if nt else 0,
        },
        "breakdown": results,
        "category_scores": category_scores,
    }


# ══════════════════════════════════════════════════════════════
# RAGAS IEEE-STANDARD EVALUATION
# ──────────────────────────────────────────────────────────────
# Computes 4 gold-standard RAG metrics:
#   • Faithfulness      – answer stays true to retrieved context
#   • Answer Relevancy  – answer actually addresses the question
#   • Context Precision – retrieved chunks are relevant (signal:noise)
#   • Context Recall    – context covers the reference answer
#
# Run on both Naive RAG and Aware RAG to directly compare
# the uplift from relationship-aware retrieval.
# ══════════════════════════════════════════════════════════════
def run_ragas_evaluation(progress_callback=None, n_questions: int = 20):
    """
    Runs RAGAS 0.4 evaluation on the first n_questions from TEST_SUITE.
    Uses Groq LLM (free) as the evaluator and HuggingFace embeddings.
    Returns a dict with scores for Naive RAG and Aware RAG side by side.
    """
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if not os.path.exists("data/chroma_db"):
        raise FileNotFoundError("Vector Database is empty. Run System Sync first.")

    # ── RAGAS imports (0.4.x API) ──────────────────────────────
    from ragas.metrics.collections import (
        Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from langchain_huggingface import HuggingFaceEmbeddings

    # ── Configure RAGAS to use free Groq LLM + local embeddings ─
    groq_llm   = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    hf_emb     = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    ragas_llm  = LangchainLLMWrapper(groq_llm)
    ragas_emb  = LangchainEmbeddingsWrapper(hf_emb)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    # Build chains
    naive_rag_chain  = create_naive_rag_chain()
    aware_rag_chain  = create_relationship_aware_rag_chain()

    suite = TEST_SUITE[:n_questions]
    total = len(suite)

    naive_samples, aware_samples = [], []

    for i, test in enumerate(suite):
        if progress_callback:
            progress_callback(i / total, f"Collecting Q{i+1}/{total} for RAGAS...")

        q   = test["query"]
        ref = test["reference"]

        # ── Naive RAG answer + contexts ─────────────────────────
        try:
            vectorstore = get_vectorstore()
            retriever   = vectorstore.as_retriever(search_kwargs={"k": 8})
            nr_docs     = retriever.invoke(q)
            nr_ctx      = [d.page_content for d in nr_docs]
            nr_ans      = naive_rag_chain.invoke({"input": q})
            naive_samples.append(SingleTurnSample(
                user_input=q,
                response=nr_ans,
                retrieved_contexts=nr_ctx,
                reference=ref,
            ))
        except Exception as e:
            naive_samples.append(SingleTurnSample(
                user_input=q, response="Error", retrieved_contexts=[""], reference=ref
            ))
        time.sleep(1)

        # ── Aware RAG answer + contexts ─────────────────────────
        try:
            aw_resp = aware_rag_chain.invoke({"input": q})
            aw_ans  = aw_resp["answer"]
            aw_docs = aw_resp.get("context", [])
            aw_ctx  = [d.page_content for d in aw_docs] if aw_docs else [""]
            aware_samples.append(SingleTurnSample(
                user_input=q,
                response=aw_ans,
                retrieved_contexts=aw_ctx,
                reference=ref,
            ))
        except Exception as e:
            aware_samples.append(SingleTurnSample(
                user_input=q, response="Error", retrieved_contexts=[""], reference=ref
            ))
        time.sleep(1)

    if progress_callback:
        progress_callback(0.5, "Running RAGAS scoring (this takes ~2 min)...")

    # ── Run RAGAS evaluate on both datasets ─────────────────────
    naive_dataset = EvaluationDataset(samples=naive_samples)
    aware_dataset = EvaluationDataset(samples=aware_samples)

    naive_result = evaluate(naive_dataset, metrics=metrics)
    aware_result = evaluate(aware_dataset, metrics=metrics)

    def extract(result):
        df = result.to_pandas()
        return {
            "faithfulness":       round(float(df["faithfulness"].mean()),       4),
            "answer_relevancy":   round(float(df["answer_relevancy"].mean()),   4),
            "context_precision":  round(float(df["context_precision"].mean()),  4),
            "context_recall":     round(float(df["context_recall"].mean()),     4),
            "ragas_score":        round(float(df[["faithfulness","answer_relevancy",
                                                   "context_precision","context_recall"]].mean(axis=1).mean()), 4),
            "per_question": df[["faithfulness","answer_relevancy",
                                "context_precision","context_recall"]].to_dict(orient="records"),
        }

    if progress_callback:
        progress_callback(1.0, "RAGAS Evaluation Complete!")

    naive_scores = extract(naive_result)
    aware_scores = extract(aware_result)

    return {
        "n_questions": n_questions,
        "naive_rag":   naive_scores,
        "aware_rag":   aware_scores,
        "questions":   [t["query"] for t in suite],
        "improvement": {
            "faithfulness":      aware_scores["faithfulness"]     - naive_scores["faithfulness"],
            "answer_relevancy":  aware_scores["answer_relevancy"] - naive_scores["answer_relevancy"],
            "context_precision": aware_scores["context_precision"]- naive_scores["context_precision"],
            "context_recall":    aware_scores["context_recall"]   - naive_scores["context_recall"],
            "ragas_score":       aware_scores["ragas_score"]      - naive_scores["ragas_score"],
        }
    }


if __name__ == "__main__":
    res = run_evaluation_suite(lambda p, m: print(f"[{p*100:.0f}%] {m}"))
    m = res["metrics"]
    print(f"\n{'='*70}")
    print(f"  LLM-AS-A-JUDGE EVALUATION  ({m['total_queries']} Questions)")
    print(f"  Scored 0-10 by LLaMA3.1 Judge | Pass threshold: 6/10")
    print(f"{'='*70}")
    print(f"  {'Model':<35} {'Avg Score':>10} {'Pass Rate':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Gemma2-9b (No Docs)':<35} {m['naive_llm_avg_score']:>9.1f}/10 {m['naive_llm_accuracy']:>9.1f}%")
    print(f"  {'Mixtral-8x7b (Basic RAG)':<35} {m['naive_rag_avg_score']:>9.1f}/10 {m['naive_rag_accuracy']:>9.1f}%")
    print(f"  {'LLaMA3 Aware RAG (Ours)':<35} {m['aware_avg_score']:>9.1f}/10 {m['aware_accuracy']:>9.1f}%")
    print(f"\n  Improvement over Gemma2:    +{m['rag_improvement_over_llm']:.1f}%")
    print(f"  Improvement over NaiveRAG:  +{m['rag_improvement_over_naive_rag']:.1f}%")
    print(f"\n  Amendment-Trap Questions ({m['tricky_total']} Qs):")
    print(f"  Gemma2-9b:    {m['tricky_naive_llm_accuracy']:.1f}%")
    print(f"  Mixtral RAG:  {m['tricky_naive_rag_accuracy']:.1f}%")
    print(f"  Aware RAG:    {m['tricky_aware_accuracy']:.1f}%")

    res = run_evaluation_suite(lambda p, m: print(f"[{p*100:.0f}%] {m}"))
    m = res["metrics"]
    print(f"\n{'='*70}")
    print(f"  LLM-AS-A-JUDGE EVALUATION  ({m['total_queries']} Questions)")
    print(f"  Scored 0-10 by LLaMA3.1 Judge | Pass threshold: 6/10")
    print(f"{'='*70}")
    print(f"  {'Model':<35} {'Avg Score':>10} {'Pass Rate':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Gemma2-9b (No Docs)':<35} {m['naive_llm_avg_score']:>9.1f}/10 {m['naive_llm_accuracy']:>9.1f}%")
    print(f"  {'Mixtral-8x7b (Basic RAG)':<35} {m['naive_rag_avg_score']:>9.1f}/10 {m['naive_rag_accuracy']:>9.1f}%")
    print(f"  {'LLaMA3 Aware RAG (Ours)':<35} {m['aware_avg_score']:>9.1f}/10 {m['aware_accuracy']:>9.1f}%")
    print(f"\n  Improvement over Gemma2:    +{m['rag_improvement_over_llm']:.1f}%")
    print(f"  Improvement over NaiveRAG:  +{m['rag_improvement_over_naive_rag']:.1f}%")
    print(f"\n  Amendment-Trap Questions ({m['tricky_total']} Qs):")
    print(f"  Gemma2-9b:    {m['tricky_naive_llm_accuracy']:.1f}%")
    print(f"  Mixtral RAG:  {m['tricky_naive_rag_accuracy']:.1f}%")
    print(f"  Aware RAG:    {m['tricky_aware_accuracy']:.1f}%")
