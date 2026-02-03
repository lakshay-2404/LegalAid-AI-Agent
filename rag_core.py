from typing import List, Tuple
import hashlib
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

    from difflib import SequenceMatcher

    best = (0.0, None)
    # Prefer .md sources for Acts (cleaner than OCR PDFs).
    candidates = list(PDF_DIR.glob("*.md"))
    if not candidates:
        candidates = list(PDF_DIR.glob("*.pdf"))

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
    queries = [query]
    query_lower = query.lower()
    
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
    }
    
    for term, synonyms in expansions.items():
        if term in query_lower:
            for syn in synonyms:
                expanded = query.lower().replace(term, syn)
                if expanded not in [q.lower() for q in queries]:
                    queries.append(expanded)
    
    return queries[:3]

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
        blocks.append(
            f"[Source {doc_number}: {source_text}]\n{d.page_content}"
        )
        doc_number += 1
    
    # Add case law sources (supporting evidence)
    for d in case_law[:max_docs_adjusted - len(blocks)]:
        meta = d.metadata
        src = ["Case Law"]
        if meta.get("act"):
            src.append(f"Act: {meta['act']}")
        if meta.get("source"):
            src.append(f"Source: {meta['source']}")
        
        source_text = " | ".join(src)
        blocks.append(
            f"[Source {doc_number}: {source_text}]\n{d.page_content}"
        )
        doc_number += 1
    
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
    
    # 3) BASE RETRIEVAL FIRST (precision-first); only expand if needed
    all_docs = {}
    doc_relevance = {}

    def merge_results(q: str) -> None:
        retrieved = vector.hybrid_retrieve(q, k=retrieval_k)
        for doc in retrieved:
            doc_id = _doc_key(doc)
            score = calculate_relevance_score(q, doc)
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
                doc_relevance[doc_id] = score
            else:
                doc_relevance[doc_id] = max(doc_relevance[doc_id], score)

    merge_results(clean_query)

    # 4) FILTER BY RELEVANCE THRESHOLD (precision-first)
    min_relevance = 0.25 if is_overview else 0.30
    relevant_docs = [doc for doc_id, doc in all_docs.items() if doc_relevance[doc_id] >= min_relevance]

    # If the query is clearly about a specific Act, try a metadata-filtered retrieval
    # to pull more of that Act for a holistic overview.
    if is_overview:
        # First: try to match the Act by filename (very reliable for "what is <act>" queries).
        target_source = detect_target_source_file(clean_query)
        if target_source:
            try:
                data = vector.vector_store._collection.get(
                    where={"source": target_source},
                    include=["documents", "metadatas"],
                    limit=1200,
                )
            except TypeError:
                try:
                    data = vector.vector_store._collection.get(
                        where={"source": target_source},
                        include=["documents", "metadatas"],
                    )
                except Exception:
                    data = None
            except Exception:
                data = None

            if data:
                for doc, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
                    if not doc:
                        continue
                    meta = meta or {}
                    d = Document(page_content=doc, metadata=meta)
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
                act_hits = vector.vector_store.similarity_search_with_score(
                    clean_query, k=retrieval_k, filter={"act": target_act}
                )
            except TypeError:
                # Some LangChain versions use `filter` on similarity_search, not similarity_search_with_score.
                act_hits = [(d, 0.0) for d in vector.vector_store.similarity_search(clean_query, k=retrieval_k, filter={"act": target_act})]
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
            data = None
            try:
                data = vector.vector_store._collection.get(
                    where={"act": target_act},
                    include=["documents", "metadatas"],
                    limit=800,
                )
            except TypeError:
                try:
                    data = vector.vector_store._collection.get(
                        where={"act": target_act},
                        include=["documents", "metadatas"],
                    )
                except Exception:
                    data = None
            except Exception:
                data = None

            if data:
                docs = []
                for doc, meta, docid in zip(
                    data.get("documents") or [],
                    data.get("metadatas") or [],
                    data.get("ids") or [],
                ):
                    if not doc:
                        continue
                    meta = meta or {}
                    if docid and "doc_id" not in meta:
                        meta["doc_id"] = docid
                    docs.append(Document(page_content=doc, metadata=meta))

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

    # Expand only if too few relevant docs (recall assist without over-expanding)
    if len(relevant_docs) < (8 if is_overview else 6):
        for q in expand_legal_query(clean_query):
            if q == clean_query:
                continue
            merge_results(q)
        relevant_docs = [doc for doc_id, doc in all_docs.items() if doc_relevance[doc_id] >= min_relevance]

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
    prompt_template = OVERVIEW_PROMPT_TEMPLATE if is_overview else PROMPT_TEMPLATE
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
