"""
RAG — the main orchestrator class for the library.

This is the only class users typically need to interact with. It wires
together a Chunker, Embedder, VectorStore, and Generator into a complete
RAG pipeline.

Lifecycle:
    rag = RAG(embedder=..., generator=...)
    rag.build_index(corpus_path="data/corpus.json")    # one-time setup
    rag.save("data/my_index.faiss", "data/chunks.json")
    # ... later ...
    rag.load("data/my_index.faiss", "data/chunks.json")
    answer = rag.query("What is backpropagation?")
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

from .chunker import Chunker
from .embedder import Embedder
from .generator import Generator
from .vector_store import VectorStore

from sentence_transformers import CrossEncoder

# Type aliases for readability
PathLike = Union[str, Path]
CorpusInput = Union[List[Dict], PathLike]


class RAG:
    """End-to-end RAG pipeline.

    Holds an embedder, generator, vector store, and chunker, and exposes
    high-level methods to build an index from a corpus, persist it, and
    query it.
    """

    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        chunker: Optional[Chunker] = None,
        vector_store: Optional[VectorStore] = None,
        top_k: int = 5,
    ):
        """
        Args:
            embedder: An Embedder instance (OpenAIEmbedder, GeminiEmbedder, etc.).
            generator: A Generator instance.
            chunker: Optional Chunker. If None, a default Chunker is created
                     with chunk_size=600 and overlap=150 (matching the
                     original rag_pipeline.py defaults).
            vector_store: Optional VectorStore. If None, an empty one is
                          created. You typically don't need to pass this in.
            top_k: Number of chunks to retrieve per query.
        """
        # Type checks: catch wrong-type arguments early with a clear error
        # rather than letting them blow up deep inside the pipeline.
        if not isinstance(embedder, Embedder):
            raise TypeError(
                f"embedder must be an Embedder instance, got {type(embedder).__name__}"
            )
        if not isinstance(generator, Generator):
            raise TypeError(
                f"generator must be a Generator instance, got {type(generator).__name__}"
            )

        self.embedder = embedder
        self.generator = generator
        self.chunker = chunker if chunker is not None else Chunker()
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        self.top_k = top_k
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ---------------------------------------------------------------------
    # Indexing
    # ---------------------------------------------------------------------

    def build_index(self, corpus: CorpusInput) -> None:
        """Build a vector index from a corpus.

        Args:
            corpus: Either a list of page dicts (with "text", "source", "page"),
                    or a path to a JSON file containing such a list.

        Side effects:
            Populates self.vector_store. Does NOT save to disk — call save()
            separately if you want persistence.
        """
        # Accept either a path or an in-memory corpus
        if isinstance(corpus, (str, Path)):
            corpus_path = Path(corpus)
            if not corpus_path.exists():
                raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)

        # 1. Chunk
        chunks = self.chunker.chunk_corpus(corpus)
        if len(chunks) == 0:
            raise ValueError(
                "Chunking produced 0 chunks. Check your corpus content and "
                "Chunker settings (especially skip_noisy_pages and min_text_length)."
            )

        # 2. Embed
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)

        # 3. Build the vector store
        self.vector_store.build(chunks, embeddings)

    def save(self, index_path: PathLike, chunks_path: PathLike) -> None:
        """Save the current index and chunks to disk.

        Args:
            index_path: Path for the FAISS index file (typically *.faiss).
            chunks_path: Path for the chunks JSON file (typically *.json).
        """
        self.vector_store.save(index_path, chunks_path)

    def load(self, index_path: PathLike, chunks_path: PathLike) -> None:
        """Load a previously-saved index and chunks from disk.

        Args:
            index_path: Path to the FAISS index file.
            chunks_path: Path to the matching chunks JSON file.
        """
        self.vector_store.load(index_path, chunks_path)

    # ---------------------------------------------------------------------
    # Querying
    # ---------------------------------------------------------------------

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Retrieve the top-k most relevant chunks for a query.

        Useful for inspection, evaluation, and debugging. Most users will
        call query() instead, which retrieves AND generates an answer.

        Args:
            query: The user's question.
            k: Number of chunks to retrieve. Defaults to self.top_k.

        Returns:
            List of chunk dicts with similarity scores.
        """
        k = k if k is not None else self.top_k

        # Embed the query (Embedder.embed takes a list, returns a list)
        query_embedding = self.embedder.embed([query])[0]

        return self.vector_store.search(query_embedding, k=k)


    def rerank_filter(self, query: str, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []

        try:
            # 1. Compute Cross-Encoder logits
            pairs = [[query, chunk["text"]] for chunk in chunks]
            scores = self.reranker.predict(pairs)

            for chunk, score in zip(chunks, scores):
                chunk["score"] = float(score)

            # 2. Sort from highest to lowest logit
            sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
            max_score = sorted_chunks[0]["score"]

            # 3. ABSOLUTE PASS: Clear Factual Matches
            if max_score >= -0.5:
                return [chunk for chunk in sorted_chunks if chunk["score"] >= -0.5]

            # 4. ABSOLUTE FAIL: Pure Garbage (No fluke match at all)
            if max_score < -4.5:
                return []

            # 5. THE DISAMBIGUATION ZONE (-4.5 < max_score < -0.5)
            # We use the top 3 chunks to calculate the Entropy Distribution
            top_k = min(3, len(sorted_chunks))
            top_scores = np.array([c["score"] for c in sorted_chunks[:top_k]])

            # Calculate Softmax (with max subtraction for numerical stability)
            exp_scores = np.exp(top_scores - np.max(top_scores))
            probs = exp_scores / np.sum(exp_scores)

            # Calculate Shannon Entropy
            entropy = -np.sum(probs * np.log(probs + 1e-9))

            # 6. DECISION GATING
            # Max entropy for k=3 is ~1.09. 
            # > 0.75 means the scores are flat (Cross-Reference: multiple partial clues)
            if entropy > 0.75:
                return sorted_chunks[:3]  # Keep top 3 for LLM synthesis

            # < 0.75 means there is a sharp dropoff (Out-of-Scope: an isolated fluke match)
            return []

        except Exception as e:
            raise RuntimeError(f"Failed to rerank chunks. Reason: {e}")

    def query(self, query: str, k: Optional[int] = None, omit_sysprompt: bool = False) -> Dict:
        """Run the full RAG pipeline: retrieve + generate.

        Args:
            query: The user's question.
            k: Number of chunks to retrieve. Defaults to self.top_k.

        Returns:
            A dict with:
                - "answer":    the generated answer string
                - "retrieved": the list of retrieved chunks (with scores)
                - "query":     the original query

            Returning a dict (not just the answer string) so the caller can
            inspect what was retrieved — important for our research benchmarks.
        """
        retrieved = self.retrieve(query, k=k)

        retrieved = self.rerank_filter(query, retrieved)

        if not retrieved:
            context = "[Database: No relevant documents found. You MUST reply with exactly: I can't answer this question.]"
        else:
            context = self._format_context(retrieved)

        answer, ppl, rmse_k, rmse_v = self.generator.generate(
            query,
            context,
            omit_sysprompt)

        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieved,
            "perplexity": ppl,
            "rmse_k": rmse_k,
            "rmse_v": rmse_v
        }

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _format_context(self, retrieved: List[Dict]) -> str:
        """Format retrieved chunks into a single context string for the generator.

        Matches the original rag_pipeline.py format so behavior is preserved.
        """
        return "\n\n".join(
            f"\n\n[Source: {c['source']} | Page {c['page']}]\n{c['text']}"
            for c in retrieved
        )

    # ---------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RAG("
            f"embedder={type(self.embedder).__name__}, "
            f"generator={type(self.generator).__name__}, "
            f"store={self.vector_store!r}, "
            f"top_k={self.top_k})"
        )
