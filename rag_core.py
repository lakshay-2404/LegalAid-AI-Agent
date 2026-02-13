from typing import List, Tuple
import hashlib
import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
# NOTE: Import the module (not individual objects) so a runtime rebuild of the
# vector store (from Streamlit) is reflected everywhere.
import vector
# from vector import hybrid_retrieve, rerank, vector_store  # previous import style (kept for reference)

STOPWORDS = {
    # Generic question scaffolding
    "what", "which", "who", "whom", "whose", "why", "how", "when", "where",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "can", "could", "would", "should", "will", "may", "might",
    "tell", "explain", "define", "meaning", "overview", "summary", "about",
    # Common function words
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by", "with", "from",
}

PDF_DIR = Path(__file__).parent / "pdfs"

# Guardrails to prevent extremely large prompts that can make local models appear to
# "think forever". These are character-based (cheap) approximations.
MAX_DOC_CHARS = 2400
MAX_CONTEXT_CHARS = 18000
MAX_CONTEXT_CHARS_OVERVIEW = 26000

def preprocess_query(query: str) -> str:
    """
    Normalize and clean query for better retrieval.
    - Remove extra whitespace
    - Convert to lowercase for processing
    - Keep important legal terms
    """
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query.strip())
    # Lowercase for consistent matching
    query = query.lower()
    # Remove special characters but keep legal punctuation
    query = re.sub(r'[^a-z0-9\s,\-./()]', '', query)
    return query

def calculate_relevance_score(query: str, doc: Document) -> float:
    """
    Calculate relevance score (0-1) for a document against query.
    Higher scores = more relevant.
    """
    score = 0.0
    text = doc.page_content.lower()
    query_lower = query.lower()
    
    # Exact phrase match (highest)
    if query_lower in text:
        score += 0.5
    
    # Query term coverage (medium)
    query_words = [w for w in query_lower.split() if len(w) > 2 and w not in STOPWORDS]
    if query_words:
        matched = sum(1 for w in query_words if w in text)
        score += (matched / len(query_words)) * 0.3
    
    # Has legal citations (bonus)
    if re.search(r'Section\s+\d+|Act:|IPC|BNS', text, re.IGNORECASE):
        score += 0.2
    
    return min(score, 1.0)  # Cap at 1.0

def deduplicate_docs(docs: List[Document], similarity_threshold: float = 0.85) -> List[Document]:
    """
    Remove duplicates and near-duplicates.

    Strategy:
    - Prefer stable `doc_id` if present.
    - Otherwise hash first 500 chars.
    - Optionally drop near-duplicates by token Jaccard similarity (cheap; n is small here).
    """
    if len(docs) <= 1:
        return docs
    
    # Use doc_id when available; fallback to content hash of first 500 chars.
    seen_hashes = {}
    unique = []
    
    for doc in docs:
        doc_id = doc.metadata.get("doc_id")
        if doc_id:
            content_hash = f"doc:{doc_id}"
        else:
            content_hash = hashlib.md5(doc.page_content[:500].lower().encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes[content_hash] = doc
            unique.append(doc)
    
    if similarity_threshold is None:
        return unique

    def tokens(s: str) -> set:
        return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 2}

    filtered: List[Document] = []
    filtered_sets: List[set] = []
    for doc in unique:
        tset = tokens(doc.page_content[:2000])
        is_dup = False
        for existing in filtered_sets:
            if not tset or not existing:
                continue
            j = len(tset & existing) / max(1, len(tset | existing))
            if j >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            filtered.append(doc)
            filtered_sets.append(tset)

    return filtered

PROMPT_TEMPLATE = """You are a legal information assistant specializing in Indian law.

PRIORITY: Accuracy and source fidelity. Use ONLY the provided legal context.

RULES:
1. Do not add facts not supported by the sources.
2. If the sources do not contain the answer, respond exactly: "Not found in retrieved sources."
3. Cite sources after EACH paragraph using the exact format: [Source N]
   - Use only the source number (do not invent citations).
   - If you cannot cite a claim, omit it.
4. Distinguish statutory provisions from case law when relevant.
5. If sources conflict, state the conflict and cite both.

LEGAL CONTEXT (prioritized by relevance):
{context}

QUESTION:
{question}

ANSWER:
"""

CLAUSE_PROMPT_TEMPLATE = """You are a legal information assistant specializing in Indian law.

PRIORITY: Accuracy and source fidelity. Use ONLY the provided legal context.

RULES:
1. Do not add facts not supported by the sources.
2. If the sources do not contain any rule relevant to assessing the clause, respond exactly: "Not found in retrieved sources."
3. Cite sources after EACH paragraph using the exact format: [Source N]
   - Use only the source number (do not invent citations).
   - If you cannot cite a claim, omit it.
4. For questions about whether a contract/employment clause is "allowed", "valid", or "enforceable":
   - Identify the relevant provision(s) in the context.
   - Explain what they say in plain language.
   - Apply them to the clause described in the question and state the conclusion.
   - Mention exceptions only if they appear in the context.

LEGAL CONTEXT (prioritized by relevance):
{context}

QUESTION:
{question}

ANSWER:
"""

OVERVIEW_PROMPT_TEMPLATE = """You are a legal information assistant specializing in Indian law.

PRIORITY: Accuracy and source fidelity. Use ONLY the provided legal context.

RULES:
1. Do not add facts not supported by the sources.
2. If the context does not identify the Act/topic at all, respond exactly: "Not found in retrieved sources."
3. Cite sources after EACH paragraph using the exact format: [Source N]
   - Use only the source number (do not invent citations).
   - If you cannot cite a claim, omit it.
4. Distinguish statutory provisions from case law examples.
5. If the question is "What is <Act>?" provide a HOLISTIC overview, not just a few sections:
   - What the Act governs (scope/purpose)
   - Who/what it applies to (applicability)
   - Core concepts and definitions (as available)
   - Formation/validity framework (as available)
   - Performance/breach/remedies framework (as available)
   - A short list of key sections (only those present in sources)
6. Organize clearly using headings: What It Is, Scope, Key Concepts, Key Sections, Practical Notes, Sources Used.
7. For scope/purpose: do NOT guess. Derive it from the sections/headings present in context and cite those sections.

LEGAL CONTEXT (prioritized by relevance):
{context}

QUESTION (Overview/Summary Query):
{question}

ANSWER:
"""

FALLBACK_PROMPT_TEMPLATE = """Provide a concise general-information answer based on your own knowledge.
You MUST:
- Start with: "Not found in retrieved sources."
- Clearly label the answer as "General knowledge (not from RAG)".
- Avoid citing sources.
- End with: "Confidence: Low. Please verify with official legal sources."

QUESTION:
{question}

ANSWER:
Not found in retrieved sources.
General knowledge (not from RAG):
"""

def issue_relevance_score(query: str, doc: Document) -> int:
    """
    Scores how likely the document answers the CORE legal issue,
    not procedural or appellate posture.
    Returns higher scores for substantive legal content.
    """
    text = doc.page_content.lower()
    score = 0

    # Positive signals for substantive issues
    substantive_terms = [
        "whether", "entitled", "right", "disability", "scribe",
        "reasonable accommodation", "examination", "candidate"
    ]

    for t in substantive_terms:
        if t in text:
            score += 2

    # Negative signals for procedural framing
    procedural_terms = [
        "high court", "appeal", "dismiss", "justified", "procedural", "jurisdiction"
    ]

    for t in procedural_terms:
        if t in text:
            score -= 3

    return score

def is_overview_query(query: str) -> bool:
    """
    Detect if query is asking for an overview/summary.
    These need broader retrieval and different prompt handling.
    """
    overview_keywords = [
        "overview", "summary", "general", "introduction", "explain",
        "what is", "describe", "elaborate", "tell me about", "about",
        "outline", "guide", "background", "understand", "meaning",
        "definition", "scope", "purpose", "key features", "main points"
    ]
    
    query_lower = query.lower()
    return any(kw in query_lower for kw in overview_keywords)


def detect_target_act(query: str, docs: List[Document]) -> str | None:
    """
    Detect the most likely Act name the user is asking about.
    Uses retrieved docs' `metadata["act"]` as candidates and chooses the best fuzzy match.
    """
    q = query.lower()
    candidates = []
    for d in docs:
        act = d.metadata.get("act")
        if act and isinstance(act, str):
            candidates.append(act)
    if not candidates:
        return None

    from difflib import SequenceMatcher

    best = (0.0, None)
    for act in set(candidates):
        score = SequenceMatcher(None, q, act.lower()).ratio()
        if score > best[0]:
            best = (score, act)
    return best[1] if best[0] >= 0.35 else None


def detect_target_source_file(query: str) -> str | None:
    """
    For "What is <Act>?" questions, filenames are often the strongest signal.
    We match against local files in pdfs/ and return the best candidate filename.
    """
    if not PDF_DIR.exists():
        return None

    q = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
    q_tokens = {t for t in q.split() if len(t) > 2 and t not in STOPWORDS}
    if not q_tokens:
        return None

    # Expand common abbreviations for better filename matching.
    if "bns" in q_tokens:
        q_tokens |= {"bharatiya", "nyaya", "sanhita"}
    if "cpc" in q_tokens:
        q_tokens |= {"civil", "procedure"}
    if "hma" in q_tokens:
        q_tokens |= {"hindu", "marriage"}
    if "ida" in q_tokens:
        q_tokens |= {"divorce"}
    if "bsa" in q_tokens or "sakshya" in q_tokens or "evidence" in q_tokens:
        q_tokens |= {"bharatiya", "sakshya", "adhiniyam"}

    # Also add abbreviations when the expanded names are present.
    if {"bharatiya", "nyaya", "sanhita"} <= q_tokens:
        q_tokens.add("bns")
    if {"bharatiya", "sakshya", "adhiniyam"} <= q_tokens:
        q_tokens.add("bsa")
    if {"civil", "procedure"} <= q_tokens:
        q_tokens.add("cpc")
    if {"hindu", "marriage"} <= q_tokens:
        q_tokens.add("hma")
    if "divorce" in q_tokens and "act" in q_tokens:
        q_tokens.add("ida")

    from difflib import SequenceMatcher

    best = (0.0, None)
    # Prefer .md/.json sources (cleaner than OCR PDFs) but include PDFs too so
    # Acts that only exist as PDFs can still be matched.
    candidates = (
        sorted(PDF_DIR.glob("*.md"))
        + sorted(PDF_DIR.glob("*.json"))
        + sorted(PDF_DIR.glob("*.pdf"))
    )

    for p in candidates:
        name = re.sub(r"[^a-z0-9]+", " ", p.stem.lower()).strip()
        name_tokens = {t for t in name.split() if len(t) > 2 and t not in STOPWORDS}
        if not name_tokens:
            continue

        overlap = len(q_tokens & name_tokens) / max(1, len(name_tokens))
        ratio = SequenceMatcher(None, q, name).ratio()
        score = (overlap * 0.7) + (ratio * 0.3)
        if score > best[0]:
            best = (score, p.name)

    return best[1] if best[0] >= 0.35 else None

def expand_legal_query(query: str) -> list:
    """Expand query with legal synonyms and related terms."""
    return expand_legal_query_with_hints(query)


def detect_issue_tags(query: str) -> set[str]:
    q = (query or "").lower()
    tags: set[str] = set()

    # Contract / clause framing
    if any(k in q for k in ["contract", "agreement", "clause", "terms", "offer letter", "bond"]):
        tags.add("contract_clause")

    # Contract-law concept queries that often omit the word "contract/act".
    if any(
        k in q
        for k in [
            "consideration",
            "offer",
            "acceptance",
            "proposal",
            "free consent",
            "coercion",
            "undue influence",
            "misrepresentation",
            "breach of contract",
            "damages",
            "compensation for breach",
            "indemnity",
            "guarantee",
        ]
    ):
        tags.add("contract_concepts")

    # Common employment/contract clause questions
    if any(
        k in q
        for k in [
            "non compete",
            "non-compete",
            "noncompete",
            "competitor",
            "same industry",
            "restraint of trade",
            "cant work",
            "cannot work",
            "work for another company",
            "join another company",
        ]
    ):
        tags.add("non_compete")

    if any(k in q for k in ["bond", "penalty", "liquidated", "damages", "forfeit", "forfeiture"]):
        tags.add("penalty_or_bond")

    # Plain-language "bond/recovery if I leave early" patterns.
    if ("training" in q or "service bond" in q) and any(k in q for k in ["pay", "recover", "recovery", "deduct", "amount", "cost"]):
        tags.add("penalty_or_bond")
    if any(k in q for k in ["leave before", "resign", "notice period"]) and any(k in q for k in ["pay", "recover", "deduct", "penalty", "bond"]):
        tags.add("penalty_or_bond")

    if any(
        k in q
        for k in [
            "sue",
            "lawsuit",
            "legal proceedings",
            "cannot go to court",
            "no court",
            "cannot sue",
            "no legal action",
            "no claim",
            "no claims",
            "cannot file case",
            "time limit to sue",
            "time limit to file",
            "waive",
            "waiver",
        ]
    ):
        tags.add("restraint_legal_proceedings")

    if any(k in q for k in ["arbitration", "arbitral", "arbitrator", "arbitrable"]):
        tags.add("arbitration")

    if any(k in q for k in ["limitation", "time barred", "time-barred", "barred by limitation"]):
        tags.add("limitation")

    if any(
        k in q
        for k in [
            "evidence",
            "admissible",
            "admissibility",
            "burden of proof",
            "presumption",
            "witness",
            "testimony",
            "cross examination",
            "cross-examination",
            "affidavit",
            "documentary evidence",
            "primary evidence",
            "secondary evidence",
            "electronic record",
            "relevancy",
            "relevance",
        ]
    ):
        tags.add("evidence")

    if any(
        k in q
        for k in [
            "cpc",
            "code of civil procedure",
            "civil procedure",
            "suit",
            "plaint",
            "written statement",
            "summons",
            "injunction",
            "stay",
            "decree",
            "execution",
            "appeal",
            "revision",
            "jurisdiction",
            "interim order",
        ]
    ):
        tags.add("civil_procedure")

    if any(
        k in q
        for k in [
            "bns",
            "ipc",
            "criminal",
            "crime",
            "offence",
            "offense",
            "fir",
            "police",
            "arrest",
            "bail",
            "chargesheet",
            "charge sheet",
            "charge-sheet",
            "punishment",
            "imprisonment",
            "fine",
            "theft",
            "cheating",
            "fraud",
            "forgery",
            "assault",
            "murder",
            "homicide",
            "kidnapping",
            "extortion",
        ]
    ):
        tags.add("criminal")

    if any(
        k in q
        for k in [
            "marriage",
            "hindu marriage",
            "hindu marriage act",
            "hma",
            "restitution of conjugal rights",
            "conjugal rights",
            "judicial separation",
            "annulment",
            "bigamy",
        ]
    ):
        tags.add("marriage")

    if any(k in q for k in ["divorce", "dissolution", "alimony", "indian divorce act"]):
        tags.add("divorce")

    if any(k in q for k in ["delimitation", "constituency", "constituencies", "electoral boundaries"]):
        tags.add("delimitation")

    if any(k in q for k in ["refund", "replacement", "defect", "defective", "deficiency", "consumer", "unfair trade", "warranty"]):
        tags.add("consumer")

    # Common civil / personal-law buckets based on available corpus
    if any(k in q for k in ["property", "sale deed", "gift deed", "lease", "rent", "tenant", "landlord", "mortgage"]):
        tags.add("property")

    if any(k in q for k in ["motor vehicle", "vehicle accident", "road accident", "driving licence", "driving license", "insurance claim"]):
        tags.add("motor_vehicle")

    if any(k in q for k in ["guardian", "guardianship", "minor custody"]):
        tags.add("guardianship")

    if any(k in q for k in ["adoption", "maintenance"]):
        tags.add("adoption_maintenance")

    if any(k in q for k in ["succession", "inheritance", "will", "probate"]):
        tags.add("succession")

    if any(k in q for k in ["specific performance", "injunction"]):
        tags.add("specific_relief")

    return tags


ISSUE_SOURCE_BOOSTS: dict[str, list[str]] = {
    # Contract/employment clauses
    "contract_clause": ["Indian Contract Act 1872.md"],
    "contract_concepts": ["Indian Contract Act 1872.md"],
    "non_compete": ["Indian Contract Act 1872.md"],
    "penalty_or_bond": ["Indian Contract Act 1872.md"],
    "restraint_legal_proceedings": ["Indian Contract Act 1872.md"],
    # Arbitration
    "arbitration": ["Arbitration and Conciliation Act 1996.md"],
    # Limitation / consumer
    "limitation": ["Limitation Act 1963.md", "Indian Contract Act 1872.md"],
    # Evidence / criminal / procedure
    "evidence": ["bsa_final.md", "indiaCodeBSA.pdf"],
    "criminal": [
        "the-bharatiya-nyaya-sanhita-2023-485731.pdf",
        "BNS2023.pdf",
        "BNS Book_After Correction.pdf",
    ],
    "civil_procedure": ["cpc.json", "the_code_of_civil_procedure,_1908.pdf", "Civil-Procedure-Code-and-Limitation-Act.pdf"],
    "consumer": ["Consumer Protection Act 2019.md", "Consumer Protection Act 1986.md"],
    # Civil/personal law buckets (available in pdfs/)
    "property": ["Transfer of Property Act 1882.md"],
    "motor_vehicle": ["Motor Vehicles Act 1988.md"],
    "guardianship": ["Hindu Minority and Guardianship Act 1956.md"],
    "adoption_maintenance": ["Hindu Adoptions and Maintenance Act 1956.md"],
    "succession": ["Indian Succession Act 1925.md"],
    "specific_relief": ["Specific Relief Act 1963.md"],
    "marriage": ["hma.json"],
    "divorce": ["ida.json", "hma.json"],
    "delimitation": ["Delimitation Act 2002.md"],
}

# "Legal knowledge layer" section injections: deterministic statutory anchors (by metadata)
# that we can pull even when user phrasing is vague. Keep these small and high-signal.
ISSUE_SECTION_INJECTIONS: dict[str, dict[str, list[str]]] = {
    "non_compete": {"Indian Contract Act 1872.md": ["27"]},
    "penalty_or_bond": {"Indian Contract Act 1872.md": ["73", "74"]},
    "restraint_legal_proceedings": {"Indian Contract Act 1872.md": ["28"]},
    "arbitration": {"Arbitration and Conciliation Act 1996.md": ["7"]},
}


def extract_section_refs(query: str) -> list[str]:
    """
    Extract section references like "section 27", "s. 27", "section 13b", "sec 10a".
    Returns normalized section IDs like ["27", "13B", "10A"].
    """
    q = (query or "").lower()
    refs: list[str] = []

    # Basic forms: section 27 / sec 27 / s. 27 / s 27
    for m in re.finditer(r"\b(?:section|sec|s)\.?\s*([0-9]{1,4}[a-z]?)\b", q):
        refs.append(m.group(1))

    # Normalize: uppercase trailing alpha suffix (e.g., 13b -> 13B)
    out: list[str] = []
    seen: set[str] = set()
    for r in refs:
        r = (r or "").strip()
        if not r:
            continue
        m = re.match(r"^([0-9]{1,4})([a-z])?$", r, flags=re.IGNORECASE)
        if not m:
            continue
        sec = m.group(1)
        suf = (m.group(2) or "").upper()
        norm = f"{sec}{suf}"
        if norm not in seen:
            out.append(norm)
            seen.add(norm)
    return out


def _collection_get_docs(where: dict, limit: int | None = None) -> List[Document]:
    """
    Read matching docs directly from the underlying vector store using metadata filters.
    This is used by the "knowledge layer" for deterministic statute retrieval.
    """
    try:
        vector.ensure_ingested()
        docs = vector.get_vector_store().get_docs(where, limit=limit)
    except Exception:
        return []

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata.setdefault("kb_injected", True)
    return docs


def knowledge_layer_injections(
    issue_tags: set[str],
    query: str,
    sources_to_boost: list[str],
    max_sections_total: int = 6,
) -> List[Document]:
    """
    Inject high-signal statutory sections based on detected issues and explicit section refs.
    """
    injected: List[Document] = []

    # 1) Issue-driven section injections (deterministic).
    section_targets: list[tuple[str, str]] = []  # (source, section)
    for tag in issue_tags:
        by_source = ISSUE_SECTION_INJECTIONS.get(tag) or {}
        for src, secs in by_source.items():
            for s in secs:
                section_targets.append((src, str(s)))

    # 2) Explicit section refs (user asked for "Section X" but may not name the Act).
    sec_refs = extract_section_refs(query)
    if sec_refs:
        src_hint = detect_target_source_file(query) or (sources_to_boost[0] if sources_to_boost else None)
        if src_hint:
            for s in sec_refs:
                section_targets.append((src_hint, s))

    # De-dupe and cap total work.
    seen_pairs: set[tuple[str, str]] = set()
    capped: list[tuple[str, str]] = []
    for src, sec in section_targets:
        key = (src, sec)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        capped.append(key)
        if len(capped) >= max_sections_total:
            break

    for src, sec in capped:
        injected.extend(_collection_get_docs({"source": src, "section": sec}, limit=6))

    return injected


def sources_for_issue_tags(tags: set[str]) -> list[str]:
    sources: list[str] = []
    for t in tags:
        sources.extend(ISSUE_SOURCE_BOOSTS.get(t, []))
    # De-dupe while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for s in sources:
        if s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def is_clause_enforceability_query(query: str) -> bool:
    """
    Heuristic for "Is this clause legal/allowed/enforceable?" type questions.
    These benefit from statutory-anchor retrieval and a clause-specific prompt.
    """
    q = (query or "").lower()

    enforceability_language = any(
        k in q
        for k in [
            "allowed",
            "legal",
            "valid",
            "enforceable",
            "binding",
            "void",
            "illegal",
            "can they",
            "can i",
        ]
    )
    if not enforceability_language:
        return False

    clause_markers = any(k in q for k in ["contract", "agreement", "clause", "terms", "bond", "offer letter", "sign"])
    tags = detect_issue_tags(q)
    implied_clause = bool(tags & {"non_compete", "penalty_or_bond", "restraint_legal_proceedings", "arbitration"})

    return clause_markers or implied_clause


def source_filtered_retrieve(query: str, source: str, k: int = 8) -> List[Document]:
    """
    Vector-only retrieval restricted to a single source file.
    Useful when user phrasing doesn't match statutory wording and global retrieval misses the key Act.
    """
    try:
        vector.ensure_ingested()
        hits = vector.get_vector_store().similarity_search_with_score(query, k=k, filter={"source": source})
    except Exception:
        return []
    return [d for d, _score in hits]


def expand_legal_query_with_hints(
    query: str,
    issue_tags: set[str] | None = None,
    max_variants: int = 6,
) -> list[str]:
    """
    Build a small set of query variants to improve recall when users use plain language.

    Design:
    - Keep the list short for performance.
    - Put high-signal statutory anchors first so they don't get truncated.
    """
    q = (query or "").strip().lower()
    if not q:
        return []

    tags = issue_tags or detect_issue_tags(q)

    anchors: list[str] = []
    # Statutory anchors (high signal)
    if "non_compete" in tags:
        anchors.extend(
            [
                "section 27 agreement in restraint of trade void indian contract act 1872",
                "agreement in restraint of trade void section 27",
            ]
        )

    if "contract_concepts" in tags:
        anchors.extend(
            [
                "indian contract act 1872 consideration offer acceptance free consent coercion undue influence misrepresentation breach damages",
            ]
        )

    if "penalty_or_bond" in tags:
        anchors.extend(
            [
                "section 74 penalty stipulated compensation breach contract indian contract act 1872",
                "section 73 compensation for breach of contract indian contract act 1872",
            ]
        )

    if "restraint_legal_proceedings" in tags:
        anchors.extend(
            [
                "section 28 agreements in restraint of legal proceedings void indian contract act 1872",
                "contract clause time limit to sue void section 28",
            ]
        )

    if "arbitration" in tags:
        anchors.extend(
            [
                "section 7 arbitration agreement arbitration and conciliation act 1996",
                "arbitration clause in a contract section 7",
            ]
        )

    if "limitation" in tags:
        anchors.extend(
            [
                "limitation act 1963 time barred period of limitation",
                "barred by limitation limitation act 1963",
            ]
        )

    if "consumer" in tags:
        anchors.extend(
            [
                "consumer protection act 2019 defect deficiency unfair trade practice",
                "consumer refund replacement warranty deficiency",
            ]
        )

    if "property" in tags:
        anchors.extend(
            [
                "transfer of property act 1882 lease rent mortgage sale gift",
            ]
        )

    if "motor_vehicle" in tags:
        anchors.extend(
            [
                "motor vehicles act 1988 accident compensation insurance",
            ]
        )

    if "guardianship" in tags:
        anchors.extend(
            [
                "hindu minority and guardianship act 1956 guardian minor custody",
            ]
        )

    if "adoption_maintenance" in tags:
        anchors.extend(
            [
                "hindu adoptions and maintenance act 1956 adoption maintenance",
            ]
        )

    if "succession" in tags:
        anchors.extend(
            [
                "indian succession act 1925 will probate succession inheritance",
            ]
        )

    if "specific_relief" in tags:
        anchors.extend(
            [
                "specific relief act 1963 specific performance injunction",
            ]
        )

    if "civil_procedure" in tags:
        anchors.extend(
            [
                "code of civil procedure cpc suit plaint written statement decree execution injunction",
                "cpc order rule decree execution",
            ]
        )

    if "evidence" in tags:
        anchors.extend(
            [
                "bharatiya sakshya adhiniyam 2023 evidence admissibility relevancy burden of proof",
                "relevancy of facts bharatiya sakshya adhiniyam 2023",
            ]
        )

    if "criminal" in tags:
        anchors.extend(
            [
                "bharatiya nyaya sanhita bns offence punishment imprisonment fine",
                "bns section punishment for offence",
            ]
        )

    if "marriage" in tags:
        anchors.extend(
            [
                "hindu marriage act 1955 marriage judicial separation restitution of conjugal rights",
            ]
        )

    if "divorce" in tags:
        anchors.extend(
            [
                "divorce act 1869 dissolution of marriage grounds",
            ]
        )

    if "delimitation" in tags:
        anchors.extend(
            [
                "delimitation act 2002 delimitation of constituencies",
            ]
        )

    # Generic synonyms (medium signal)
    expansions = {
        "punishment": ["penalty", "sentence", "imprisonment"],
        "offence": ["offense", "crime", "criminal act"],
        "section": ["article", "provision", "clause"],
        "act": ["law", "statute", "legislation"],
        "court": ["tribunal", "judicial body"],
        "judgment": ["verdict", "ruling", "decision", "holding"],
        "culpable homicide": ["murder", "killing", "manslaughter"],
        "bns": ["bharatiya nyaya sanhita", "indian penal code"],
        "ipc": ["indian penal code", "penal code"],
        "right": ["privilege", "entitlement", "liberty"],
        "liability": ["responsibility", "legal obligation"],
        # Employment/non-compete phrasing often doesn't match statutory wording.
        "non compete": ["non-compete", "restraint of trade"],
        "non-compete": ["non compete", "restraint of trade"],
        "noncompete": ["non compete", "restraint of trade"],
        "cant work": ["cannot work", "restraint of trade"],
        "cannot work": ["cant work", "restraint of trade"],
        "same industry": ["restraint of trade", "trade restriction"],
        # Bonds / penalties
        "bond": ["penalty", "liquidated damages", "compensation for breach"],
        "liquidated damages": ["penalty", "compensation for breach"],
        # Courts / suing
        "sue": ["legal proceedings", "court"],
        "lawsuit": ["legal proceedings", "court"],
        # Evidence act phrasing
        "evidence": ["sakshya", "bharatiya sakshya adhiniyam", "relevancy of facts"],
        "admissible": ["admissibility", "relevant", "relevancy"],
        "burden": ["burden of proof"],
        # CPC phrasing
        "injunction": ["temporary injunction", "interim order"],
        "decree": ["execution of decree"],
        # Family law
        "divorce": ["dissolution of marriage", "judicial separation", "alimony"],
    }

    variants: list[str] = [q]
    variants.extend(anchors)

    for term, synonyms in expansions.items():
        if term in q:
            for syn in synonyms:
                # Prefer short, high-signal variants.
                replaced = q.replace(term, syn)
                if replaced != q:
                    variants.append(replaced)
                variants.append(syn)

    # De-duplicate while preserving order, then truncate.
    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        v = (v or "").strip().lower()
        if not v or v in seen:
            continue
        out.append(v)
        seen.add(v)
        if len(out) >= max_variants:
            break

    return out

def estimate_context_confidence(docs: List[Document]) -> float:
    """Estimate confidence in retrieved context (0-1)."""
    if not docs:
        return 0.0
    
    confidence = 0.0
    
    # Bonus for statutory sources
    statutory_count = sum(
        1 for d in docs if d.metadata.get("doc_type") in {"pdf", "md", "json"}
    )
    confidence += min(statutory_count / len(docs), 1.0) * 0.4
    
    # Bonus for specific citations
    cited_count = sum(1 for d in docs if d.metadata.get("section"))
    confidence += min(cited_count / len(docs), 1.0) * 0.3
    
    # Bonus for multiple sources
    confidence += min(len(docs) / 5.0, 1.0) * 0.3
    
    return min(confidence, 1.0)

def _doc_key(doc: Document) -> str:
    doc_id = doc.metadata.get("doc_id")
    if doc_id:
        return f"doc:{doc_id}"
    return hashlib.md5(doc.page_content[:500].lower().encode()).hexdigest()


def fallback_answer(query: str, model: OllamaLLM) -> str:
    prompt = FALLBACK_PROMPT_TEMPLATE.format(question=query)
    return model.invoke(prompt)


def annotate_answer(answer: str, confidence: float, provenance: str) -> str:
    """
    Always add a provenance/confidence footer for user trust.
    """
    if confidence >= 0.8:
        conf_line = "Confidence: High"
    elif confidence >= 0.6:
        conf_line = "Confidence: Medium"
    elif confidence >= 0.4:
        conf_line = "Confidence: Limited"
    else:
        conf_line = "Confidence: Low"

    return f"{answer}\n\n---\n\nProvenance: {provenance}\n{conf_line}"


def rag_answer_looks_grounded(answer: str) -> bool:
    # Accept either "[Source 1]" or "[Source 1: ...]".
    return bool(re.search(r"\[Source\s+\d+\s*[:\]]", answer))


def citation_coverage(answer: str) -> float:
    """
    Rough heuristic: fraction of non-empty paragraphs that contain at least one [Source N] marker.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", answer) if p.strip()]
    if not paras:
        return 0.0
    cited = sum(1 for p in paras if re.search(r"\[Source\s+\d+\s*[:\]]", p))
    return cited / len(paras)

def build_context(
    docs: List[Document],
    max_docs: int = 10,
    is_overview: bool = False,
    query: str | None = None,
    target_act: str | None = None,
) -> Tuple[str, float]:
    """
    Build context with better organization and legal metadata.
    Prioritizes statutory sources over case law.
    For overview queries, includes more comprehensive content.
    Returns (context_text, confidence_score)
    """
    if not docs:
        return "", 0.0
    
    # For overview questions, use even more documents for better coverage
    max_docs_adjusted = max_docs * 2 if is_overview else max_docs
    
    # Deduplicate documents
    docs = deduplicate_docs(docs[:max_docs_adjusted])
    
    # Separate statutory and case law
    statutory = [d for d in docs if d.metadata.get("doc_type") in {"pdf", "md", "json"}]
    case_law = [d for d in docs if d.metadata.get("doc_type") == "qa"]

    # For "What is <Act>?" style overview queries, prefer more chunks from that Act and
    # prioritize definitional/structural sections so the answer is holistic.
    if is_overview and target_act:
        act_statutory = [d for d in statutory if (d.metadata.get("act") or "").lower() == target_act.lower()]
        if act_statutory:
            statutory = act_statutory

        keywords = [
            "short title", "extent", "commencement", "application",
            "definitions", "interpretation",
            "contract", "agreement", "consideration", "proposal", "acceptance",
            "competent", "free consent", "void", "voidable", "unlawful",
            "performance", "breach", "damages", "compensation", "remedy",
            "indemnity", "guarantee",
        ]

        q_terms = []
        if query:
            q_terms = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) > 2 and t not in STOPWORDS]

        def section_num(d: Document) -> int:
            s = d.metadata.get("section")
            try:
                return int(str(s).strip())
            except Exception:
                return 10**9

        def overview_score(d: Document) -> int:
            txt = d.page_content.lower()
            score = 0
            for kw in keywords:
                if kw in txt:
                    score += 5
            for t in q_terms:
                if t in txt:
                    score += 2
            # Prefer early, structural sections if we can parse them.
            n = section_num(d)
            if n < 50:
                score += 3
            if n < 10:
                score += 3
            return score

        # Pick a mix: high-signal sections + early structural sections for breadth.
        ranked = sorted(statutory, key=lambda d: (overview_score(d), -len(d.page_content)), reverse=True)
        early = sorted(statutory, key=lambda d: (d.metadata.get("section") is None, section_num(d)))

        chosen: List[Document] = []
        seen = set()
        for d in ranked[:max_docs_adjusted]:
            k = _doc_key(d)
            if k in seen:
                continue
            chosen.append(d)
            seen.add(k)
        for d in early[: min(8, max_docs_adjusted)]:
            k = _doc_key(d)
            if k in seen:
                continue
            chosen.append(d)
            seen.add(k)

        statutory = chosen[:max_docs_adjusted]
    
    blocks = []
    doc_number = 1
    max_context_chars = MAX_CONTEXT_CHARS_OVERVIEW if is_overview else MAX_CONTEXT_CHARS
    context_chars = 0

    def _add_block(source_text: str, content: str) -> None:
        nonlocal doc_number, context_chars
        if context_chars >= max_context_chars:
            return
        content = (content or "").strip()
        if not content:
            return
        if len(content) > MAX_DOC_CHARS:
            content = content[:MAX_DOC_CHARS].rstrip() + "\n...[truncated]"
        block = f"[Source {doc_number}: {source_text}]\n{content}"
        # Roughly account for separators and headers.
        projected = context_chars + len(block) + 10
        if projected > max_context_chars and context_chars > 0:
            return
        blocks.append(block)
        context_chars = projected
        doc_number += 1
    
    # Add statutory sources first (higher authority) - prioritize for overviews
    for d in statutory[:max_docs_adjusted // 2 + 3 if is_overview else max_docs_adjusted // 2 + 2]:
        meta = d.metadata
        src = []
        if meta.get("citation"):
            src.append(f"**{meta['citation']}**")
        elif meta.get("act"):
            src.append(f"Act: {meta['act']}")
        if meta.get("section") and not meta.get("citation"):
            src.append(f"Section: {meta['section']}")
        if meta.get("chapter"):
            src.append(f"Chapter: {meta['chapter']}")
        
        source_text = " | ".join(src) if src else meta.get("source", "Unknown")
        _add_block(source_text, d.page_content)
    
    # Add case law sources (supporting evidence)
    for d in case_law[:max_docs_adjusted - len(blocks)]:
        if context_chars >= max_context_chars:
            break
        meta = d.metadata
        src = ["Case Law"]
        if meta.get("act"):
            src.append(f"Act: {meta['act']}")
        if meta.get("source"):
            src.append(f"Source: {meta['source']}")
        
        source_text = " | ".join(src)
        _add_block(source_text, d.page_content)
    
    context = "\n\n---\n\n".join(blocks)
    confidence = estimate_context_confidence(docs)
    
    return context, confidence

def case_consistency_score(query: str, doc: Document) -> int:
    """Score document relevance for case-law queries."""
    q = query.lower()
    d = doc.page_content.lower()
    score = 0

    # Case name tokens
    for t in q.split():
        if len(t) > 3 and t in d:
            score += 2

    # Penalize vague admin language (noise pattern)
    if "non-examinable" in d:
        score -= 5

    return score

def answer_query(
    query: str,
    model: OllamaLLM,
) -> Tuple[str, List[Document]]:
    """
    Complete RAG pipeline with multi-stage retrieval and confidence scoring.
    Handles both specific queries and overview/summary questions.
    """
    
    # Input validation - prevent prompt injection and cache poisoning
    if not query or not query.strip():
        return "Please provide a valid question.", []
    
    # Limit query length to prevent abuse
    if len(query) > 1000:
        return "Question is too long. Please keep it under 1000 characters.", []
    
    # Check for obvious prompt injection patterns
    injection_patterns = [
        r"ignore.*instruction",
        r"system.*prompt",
        r"disregard.*rule",
        r"override.*constraint"
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return "Query rejected: Invalid pattern detected.", []

    # 1️⃣ DETECT QUERY TYPE
    is_overview = is_overview_query(query)
    retrieval_k = 60 if is_overview else 30  # More context for Act overview queries
    
    # 2️⃣ PREPROCESS QUERY
    clean_query = preprocess_query(query)

    issue_tags = detect_issue_tags(clean_query)
    clause_query = is_clause_enforceability_query(clean_query)
    issue_force = bool(issue_tags - {"contract_clause"})

    # Plain-language clause/issue questions benefit from broader retrieval.
    retrieval_k = 60 if is_overview else (45 if clause_query or issue_force or "contract_clause" in issue_tags else 30)
    max_variants = 8 if len(issue_tags) >= 2 else 6
    query_variants = expand_legal_query_with_hints(clean_query, issue_tags=issue_tags, max_variants=max_variants)
    sources_to_boost = sources_for_issue_tags(issue_tags)

    # If the query mentions an Act by name (or close to it), boost that source directly.
    target_source_hint = detect_target_source_file(clean_query)
    if target_source_hint and target_source_hint not in sources_to_boost:
        sources_to_boost = [target_source_hint] + sources_to_boost
    
    # 3) BASE RETRIEVAL FIRST (precision-first); only expand if needed
    all_docs = {}
    doc_relevance = {}

    def merge_docs(retrieved: List[Document], q: str, score_floor: float | None = None) -> None:
        for doc in retrieved:
            doc_id = _doc_key(doc)
            score = calculate_relevance_score(q, doc)
            if score_floor is not None:
                score = max(score, score_floor)
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
                doc_relevance[doc_id] = score
            else:
                doc_relevance[doc_id] = max(doc_relevance[doc_id], score)

    def merge_results(q: str) -> None:
        merge_docs(vector.hybrid_retrieve(q, k=retrieval_k), q)

    merge_results(clean_query)

    # 4) FILTER BY RELEVANCE THRESHOLD (precision-first)
    # Clause/issue queries often use plain language, so use a slightly lower threshold.
    min_relevance = 0.25 if (is_overview or clause_query or issue_force) else 0.30

    # Graph-augmented vector retrieval (optional): use Neo4j traversal to restrict Milvus search to a
    # high-signal candidate set (Act/Section neighborhood).
    if os.environ.get("ENABLE_GRAPH", "0").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from orchestrator import get_orchestrator

            gdocs = get_orchestrator().graph_vector_retrieve(clean_query, k=min(retrieval_k, 20))
            if gdocs:
                # Give a small relevance floor to keep graph hits in the pool.
                merge_docs(gdocs, clean_query, score_floor=max(min_relevance, 0.35))
        except Exception:
            pass

    # Knowledge-layer injection: deterministically pull key statutory sections (by metadata) when possible.
    kb_docs = knowledge_layer_injections(issue_tags, clean_query, sources_to_boost, max_sections_total=6)
    if kb_docs:
        # Give injected docs a relevance floor so they aren't dropped by heuristics.
        merge_docs(kb_docs, clean_query, score_floor=max(min_relevance + 0.05, 0.45))

    relevant_docs = [doc for doc_id, doc in all_docs.items() if doc_relevance[doc_id] >= min_relevance]

    # If the query is clearly about a specific Act, try a metadata-filtered retrieval
    # to pull more of that Act for a holistic overview.
    if is_overview:
        # First: try to match the Act by filename (very reliable for "what is <act>" queries).
        target_source = detect_target_source_file(clean_query)
        if target_source:
            for d in _collection_get_docs({"source": target_source}, limit=1200):
                doc_id = _doc_key(d)
                score = calculate_relevance_score(clean_query, d)
                if doc_id not in all_docs:
                    all_docs[doc_id] = d
                    doc_relevance[doc_id] = score
                else:
                    doc_relevance[doc_id] = max(doc_relevance[doc_id], score)

        # Recompute relevant docs after filename/act enrichment (important).
        relevant_docs = [doc for doc_id, doc in all_docs.items() if doc_relevance[doc_id] >= min_relevance]

        # Second: detect target act from candidates (either initial retrieval or filename pull).
        target_act = detect_target_act(clean_query, list(all_docs.values()))
        if target_act:
            try:
                vector.ensure_ingested()
                act_hits = vector.get_vector_store().similarity_search_with_score(clean_query, k=retrieval_k, filter={"act": target_act})
            except Exception:
                act_hits = []

            for d, _dist in act_hits:
                doc_id = _doc_key(d)
                score = calculate_relevance_score(clean_query, d)
                if doc_id not in all_docs:
                    all_docs[doc_id] = d
                    doc_relevance[doc_id] = score
                else:
                    doc_relevance[doc_id] = max(doc_relevance[doc_id], score)

            # Also fetch an "outline" set from that Act using metadata-only access.
            # This helps overview queries pull in definitions/scope sections that embeddings may miss.
            docs = _collection_get_docs({"act": target_act}, limit=800)
            if docs:

                def sec_num(d: Document) -> int:
                    s = d.metadata.get("section")
                    try:
                        return int(str(s).strip())
                    except Exception:
                        return 10**9

                key_terms = [
                    "short title", "extent", "commencement", "application",
                    "definitions", "interpretation",
                    "contract", "agreement", "consideration", "proposal", "acceptance",
                    "competent", "free consent", "void", "voidable", "unlawful",
                    "performance", "breach", "damages", "compensation", "remedy",
                    "indemnity", "guarantee",
                ]

                def outline_score(d: Document) -> int:
                    txt = d.page_content.lower()
                    score = 0
                    for kw in key_terms:
                        if kw in txt:
                            score += 3
                    n = sec_num(d)
                    if n < 20:
                        score += 4
                    if n < 10:
                        score += 3
                    return score

                docs = sorted(docs, key=lambda d: (outline_score(d), -len(d.page_content)), reverse=True)[:25]
                for d in docs:
                    doc_id = _doc_key(d)
                    score = calculate_relevance_score(clean_query, d)
                    if doc_id not in all_docs:
                        all_docs[doc_id] = d
                        doc_relevance[doc_id] = score
                    else:
                        doc_relevance[doc_id] = max(doc_relevance[doc_id], score)

    # Recall/rescue: plain-language legal queries often miss the exact statutory wording.
    ordered_ids = sorted(all_docs.keys(), key=lambda i: doc_relevance.get(i, 0.0), reverse=True)
    top_docs = [all_docs[i] for i in ordered_ids[:12]]
    base_conf = estimate_context_confidence(top_docs)
    base_statutory = sum(1 for d in top_docs if d.metadata.get("doc_type") in {"pdf", "md", "json"})
    needs_help = (base_conf < (0.50 if is_overview else 0.45)) or (base_statutory < (3 if is_overview else 2))

    should_expand = (
        clause_query
        or issue_force
        or needs_help
        or (len(relevant_docs) < (8 if is_overview else 6))
    )

    if should_expand:
        for qv in query_variants[1:4]:
            if qv and qv != clean_query:
                merge_results(qv)

        # If we can infer likely Acts, do a source-restricted vector search for extra recall.
        if sources_to_boost and (clause_query or issue_force or needs_help):
            boost_queries = query_variants[:3] or [clean_query]
            for src in sources_to_boost[:3]:
                for qv in boost_queries:
                    merge_docs(source_filtered_retrieve(qv, src, k=8), qv)

        relevant_docs = [doc for doc_id, doc in all_docs.items() if doc_relevance[doc_id] >= min_relevance]
        ordered_ids = sorted(all_docs.keys(), key=lambda i: doc_relevance.get(i, 0.0), reverse=True)

    # Always keep a small pool even if relevance heuristics are pessimistic.
    if all_docs:
        max_keep = 90 if is_overview else 60
        min_keep = 18 if is_overview else 12

        filtered_ids = [i for i in ordered_ids if doc_relevance.get(i, 0.0) >= min_relevance]
        pool_ids = filtered_ids if len(filtered_ids) >= min_keep else ordered_ids[:min_keep]

        # Ensure we have at least a couple statutory sources when available.
        statutory_in_pool = sum(
            1 for i in pool_ids if all_docs[i].metadata.get("doc_type") in {"pdf", "md", "json"}
        )
        if statutory_in_pool < 2:
            for i in ordered_ids:
                if i in pool_ids:
                    continue
                if all_docs[i].metadata.get("doc_type") in {"pdf", "md", "json"}:
                    pool_ids.append(i)
                    statutory_in_pool += 1
                    if statutory_in_pool >= 2:
                        break

        relevant_docs = [all_docs[i] for i in pool_ids[:max_keep]]

    # 5) DEDUPLICATE DOCUMENTS
    relevant_docs = deduplicate_docs(relevant_docs, similarity_threshold=0.85)
    
    # 6️⃣ CONTEXT-AWARE SCORING & RERANKING
    if "issue" in clean_query.lower():
        relevant_docs = sorted(
            relevant_docs,
            key=lambda d: issue_relevance_score(clean_query, d),
            reverse=True
        )
    elif " vs " in clean_query.lower():
        relevant_docs = sorted(
            relevant_docs,
            key=lambda d: case_consistency_score(clean_query, d),
            reverse=True
        )
    
    # Apply combined relevance+authority reranking
    relevant_docs = vector.rerank(relevant_docs, query=clean_query)
    
    # 7️⃣ CHECK FOR SUFFICIENT CONTEXT
    if not relevant_docs:
        answer = fallback_answer(query, model)
        return annotate_answer(answer, 0.2, "General knowledge (not from RAG)"), []
    
    # 8️⃣ BUILD CONTEXT WITH CONFIDENCE (adjusted for overview)
    target_act = detect_target_act(clean_query, relevant_docs) if is_overview else None
    context, confidence = build_context(
        relevant_docs,
        max_docs=15 if is_overview else 10,
        is_overview=is_overview,
        query=clean_query,
        target_act=target_act,
    )
    
    if not context.strip():
        answer = fallback_answer(query, model)
        return annotate_answer(answer, 0.2, "General knowledge (not from RAG)"), []
    
    # 9️⃣ GENERATE ANSWER (use appropriate prompt)
    if is_overview:
        prompt_template = OVERVIEW_PROMPT_TEMPLATE
    elif clause_query:
        prompt_template = CLAUSE_PROMPT_TEMPLATE
    else:
        prompt_template = PROMPT_TEMPLATE
    prompt = prompt_template.format(
        context=context,
        question=query,
    )

    answer = model.invoke(prompt)

    # Citation hygiene: prefer per-paragraph citations for precision.
    if not rag_answer_looks_grounded(answer):
        confidence = min(confidence, 0.35)
        answer = answer + "\n\nNote: No [Source N] citations were included; treat as low confidence."
    else:
        cov = citation_coverage(answer)
        if cov < 0.5:
            answer = (
                answer
                + "\n\nNote: Citations were provided but not consistently per paragraph; please verify against the cited sources."
            )

    # If the model says the answer isn't in the retrieved context, fall back to general knowledge
    # (but label it clearly as not-from-RAG and low confidence).
    if answer.strip().startswith("Not found in retrieved sources."):
        fb = fallback_answer(query, model)
        return annotate_answer(fb, 0.2, "General knowledge (not from RAG)"), []

    return annotate_answer(answer, confidence, "Retrieved sources (RAG)"), relevant_docs[:5]
