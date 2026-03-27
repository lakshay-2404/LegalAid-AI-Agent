"""
Parent section reconstruction utilities.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from langchain_core.documents import Document


def group_by_section(docs: List[Document]) -> Dict[Tuple[str, str, str], List[Document]]:
    buckets: Dict[Tuple[str, str, str], List[Document]] = defaultdict(list)
    for d in docs:
        m = d.metadata or {}
        key = (m.get("source_path"), m.get("act"), m.get("section"))
        if any(key):
            buckets[key].append(d)
    return buckets


def reconstruct_sections(docs: List[Document]) -> List[Document]:
    grouped = group_by_section(docs)
    out: List[Document] = []
    seen = set()
    for d in docs:
        m = d.metadata or {}
        key = (m.get("source_path"), m.get("act"), m.get("section"))
        if key in grouped and key not in seen:
            siblings = grouped[key]
            siblings_sorted = sorted(
                siblings,
                key=lambda x: (
                    (x.metadata or {}).get("paragraph_number")
                    or (x.metadata or {}).get("char_count", 0)
                    or 0
                ),
            )
            text = "\n\n".join(doc.page_content for doc in siblings_sorted if doc.page_content)
            parent_meta = dict(m)
            parent_meta["parent_reconstructed"] = True
            out.append(Document(page_content=text, metadata=parent_meta))
            seen.add(key)
        elif key not in grouped or not any(key):
            out.append(d)
    return out
