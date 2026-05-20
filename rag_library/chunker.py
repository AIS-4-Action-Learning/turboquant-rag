"""
Chunker — splits text into overlapping chunks suitable for embedding.

Takes a corpus (list of pages with source/page metadata) and produces a list
of chunks ready to be embedded. Tries to split on natural boundaries
(newlines, sentence ends) rather than mid-word.

Migrated from the original rag_pipeline.py functions:
    - clean_text
    - is_noisy_page
    - find_split_point
    - chunk_page (was buggy in the original — referenced as `chunk_page` but
                  only defined as `chunk_page_old`; that bug is fixed here)
    - create_chunks
"""

import uuid
from typing import Dict, List


class Chunker:
    """Splits a corpus of text pages into overlapping chunks.

    A "corpus" is a list of dicts, each with keys:
        - "text":   the page's text content
        - "source": filename or identifier of the document
        - "page":   page number within the document

    The output is a list of chunk dicts, each with:
        - "chunk_id":    unique UUID for the chunk
        - "source":      original document name
        - "page":        original page number
        - "chunk_index": position of this chunk within its page
        - "text":        the chunk text
        - "char_count":  length in characters
    """

    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 150,
        min_text_length: int = 100,
        skip_noisy_pages: bool = True,
    ):
        """
        Args:
            chunk_size: Target chunk size in characters.
            overlap: Number of characters of overlap between consecutive chunks.
                     Helps preserve context across chunk boundaries.
            min_text_length: Pages shorter than this are considered noise and
                             skipped (e.g., blank pages, cover pages).
            skip_noisy_pages: If True, drops pages flagged as noisy
                              (TOC pages, mostly-empty pages). Set False to
                              keep all pages regardless.
        """
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than chunk_size "
                f"({chunk_size}); otherwise chunks would loop indefinitely."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_text_length = min_text_length
        self.skip_noisy_pages = skip_noisy_pages

    # ----- public API -----

    def chunk_corpus(self, corpus: List[Dict]) -> List[Dict]:
        """Chunk an entire corpus.

        Args:
            corpus: List of page dicts, each with "text", "source", "page".

        Returns:
            Flat list of chunk dicts across all pages.
        """
        all_chunks = []

        for entry in corpus:
            text = entry["text"]
            source = entry["source"]
            page = entry["page"]

            if self.skip_noisy_pages and self._is_noisy_page(text):
                continue

            page_chunks = self._chunk_page(text, source, page)
            all_chunks.extend(page_chunks)

        return all_chunks

    # ----- internal helpers -----

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and strip leading/trailing space."""
        text = text.strip()
        text = " ".join(text.split())
        return text

    def _is_noisy_page(self, text: str) -> bool:
        """Heuristic: detect pages that are unlikely to contain useful content.

        Catches: very short pages, table-of-contents pages, dot-leader patterns.
        These heuristics were tuned for the original Red Cross PDFs; they may
        need to be adjusted (or disabled via skip_noisy_pages=False) for
        other corpora like the deep learning textbook.
        """
        if len(text) < self.min_text_length:
            return True
        if "table of contents" in text.lower():
            return True
        if text.count("...") > 5:  # typical TOC dot-leader pattern
            return True
        return False

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """Find the best place to end a chunk between [start, end).

        Prefers (in order): newline, sentence-ending period, then the hard
        end position if no natural boundary is found.
        """
        newline_pos = text.rfind("\n", start, end)
        period_pos = text.rfind(".", start, end)

        split_pos = max(newline_pos, period_pos)

        if split_pos > start:
            return split_pos + 1
        return end

    def _chunk_page(
        self, text: str, source: str, page_number: int
    ) -> List[Dict]:
        """Chunk a single page into overlapping pieces.

        This is the function the original code intended to call as `chunk_page`
        but accidentally named `chunk_page_old` — fixed here.
        """
        chunks = []
        cursor = 0
        chunk_index = 0

        text = self._clean_text(text)

        while cursor < len(text):
            end = min(cursor + self.chunk_size, len(text))

            # Adjust split point to a natural boundary if possible
            if end < len(text):
                end = self._find_split_point(text, cursor, end)

            chunk_text = text[cursor:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "source": source,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "char_count": len(chunk_text),
                    }
                )
                chunk_index += 1

            cursor += self.chunk_size - self.overlap

        return chunks
