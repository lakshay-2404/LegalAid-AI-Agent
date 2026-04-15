"""
Parent section reconstruction utilities.
"""
from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document


SectionKey = Tuple[str, str, str]


def _clean_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _source_key(metadata: dict) -> str:
    source_path = _clean_str(metadata.get("source_path"))
    if source_path:
        return source_path

    doc_id = _clean_str(metadata.get("doc_id"))
    if doc_id.count(":") >= 2:
        return doc_id.rsplit(":", 2)[0]

    source = _clean_str(metadata.get("source"))
    if source:
        return source

    return ""


def _section_key(metadata: dict) -> Optional[SectionKey]:
    source = _source_key(metadata)
    act = _clean_str(metadata.get("act"))
    section = _clean_str(metadata.get("section"))
    if not (source and act and section):
        return None
    return (source, act, section)


def _as_int(value: object) -> Optional[int]:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _doc_chunk_index(metadata: dict) -> Optional[int]:
    doc_id = _clean_str(metadata.get("doc_id"))
    if doc_id.count(":") < 2:
        return None
    return _as_int(doc_id.rsplit(":", 2)[1])


def _sort_position(metadata: dict) -> Tuple[int, int]:
    paragraph_number = _as_int(metadata.get("paragraph_number"))
    if paragraph_number is not None:
        return (0, paragraph_number)

    paragraph_start = _as_int(metadata.get("paragraph_start"))
    if paragraph_start is not None:
        return (1, paragraph_start)

    chunk_index = _doc_chunk_index(metadata)
    if chunk_index is not None:
        return (2, chunk_index)

    return (3, 0)


def group_by_section(docs: Iterable[Document]) -> dict[SectionKey, list[Document]]:
    buckets: DefaultDict[SectionKey, list[Document]] = defaultdict(list)
    for doc in docs:
        metadata = doc.metadata or {}
        key = _section_key(metadata)
        if key is not None:
            buckets[key].append(doc)
    return dict(buckets)


def _merge_siblings(siblings: List[Document], parent_metadata: dict) -> Document:
    siblings_sorted = [
        doc
        for _, doc in sorted(
            enumerate(siblings),
            key=lambda item: (*_sort_position(item[1].metadata or {}), item[0]),
        )
    ]
    text = "\n\n".join(doc.page_content for doc in siblings_sorted if doc.page_content)
    merged_metadata = dict(parent_metadata)
    merged_metadata["parent_reconstructed"] = True
    return Document(page_content=text, metadata=merged_metadata)


def reconstruct_sections(
    docs: List[Document],
    all_docs: Optional[Iterable[Document]] = None,
) -> List[Document]:
    grouped = group_by_section(all_docs if all_docs is not None else docs)
    out: List[Document] = []
    seen: set[SectionKey] = set()
    for doc in docs:
        metadata = doc.metadata or {}
        key = _section_key(metadata)
        if key is None or key not in grouped:
            out.append(doc)
            continue
        if key in seen:
            continue
        out.append(_merge_siblings(grouped[key], metadata))
        seen.add(key)
    return out
