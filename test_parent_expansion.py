import unittest

from langchain_core.documents import Document

from parent_expansion import reconstruct_sections


def _doc(
    text: str,
    *,
    source_path: str = "acts/sample.md",
    act: str | None = "Sample Act",
    section: str | None = "12",
    doc_id: str | None = None,
    **metadata,
) -> Document:
    meta = {"source_path": source_path}
    if act is not None:
        meta["act"] = act
    if section is not None:
        meta["section"] = section
    if doc_id is not None:
        meta["doc_id"] = doc_id
    meta.update(metadata)
    return Document(page_content=text, metadata=meta)


class ReconstructSectionsTests(unittest.TestCase):
    def test_skips_sectionless_chunks(self) -> None:
        docs = [
            _doc("Heading chunk", act=None, section=None),
            _doc("Preamble chunk", act=None, section=None),
        ]

        reconstructed = reconstruct_sections(docs)

        self.assertEqual(reconstructed, docs)
        self.assertTrue(all(not d.metadata.get("parent_reconstructed") for d in reconstructed))

    def test_orders_siblings_by_doc_id_when_paragraphs_missing(self) -> None:
        later_chunk = _doc(
            "Second chunk",
            doc_id="acts/sample.md:1:deadbeef",
        )
        earlier_chunk = _doc(
            "First chunk with more text to avoid any accidental length-based ordering",
            doc_id="acts/sample.md:0:deadbeef",
        )

        reconstructed = reconstruct_sections([later_chunk, earlier_chunk])

        self.assertEqual(len(reconstructed), 1)
        self.assertEqual(
            reconstructed[0].page_content,
            "First chunk with more text to avoid any accidental length-based ordering\n\n"
            "Second chunk",
        )
        self.assertIs(reconstructed[0].metadata["parent_reconstructed"], True)

    def test_expands_selected_docs_from_full_doc_pool(self) -> None:
        selected = [
            _doc(
                "Second chunk",
                doc_id="acts/sample.md:1:deadbeef",
            )
        ]
        all_docs = [
            _doc(
                "First chunk",
                doc_id="acts/sample.md:0:deadbeef",
            ),
            selected[0],
        ]

        reconstructed = reconstruct_sections(selected, all_docs)

        self.assertEqual(len(reconstructed), 1)
        self.assertEqual(reconstructed[0].page_content, "First chunk\n\nSecond chunk")
        self.assertIs(reconstructed[0].metadata["parent_reconstructed"], True)


if __name__ == "__main__":
    unittest.main()
