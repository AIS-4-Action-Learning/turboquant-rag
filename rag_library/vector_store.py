"""
VectorStore — wraps FAISS index for indexing and similarity search.

Holds an in-memory FAISS index plus the chunks it was built from. Provides
methods to build the index from chunks, save it to disk, load it from disk,
and search for the top-k most similar chunks given a query embedding.

Migrated from the original rag_pipeline.py functions:
    - build_index
    - load_index
    - retrieve

Note: this class is responsible for vector storage and similarity search.
The actual embedding (text → vector) is handled by Embedder. Keeping these
separate means we can swap embedders (OpenAI, Llama BF16, Llama TurboQuant)
without touching the index code.
"""

import json
from pathlib import Path
from typing import Dict, List, Union

import faiss
import numpy as np

# Type alias: a path can be given as either a string or a Path object.
PathLike = Union[str, Path]


class VectorStore:
    """A FAISS-backed vector store for chunk embeddings.

    Uses inner-product similarity on L2-normalized vectors, which is
    equivalent to cosine similarity. This matches the retrieval behavior
    of the original rag_pipeline.py.

    Lifecycle:
        store = VectorStore()
        store.build(chunks, embeddings)         # build new index
        store.save(index_path, chunks_path)     # persist to disk
        store.load(index_path, chunks_path)     # restore from disk
        results = store.search(query_embedding, k=5)
    """

    def __init__(self):
        """Create an empty vector store. Call build() or load() to populate."""
        self.index = None
        self.chunks = None

    # ----- index construction -----

    def build(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        """Build a FAISS index from chunks and their embeddings.

        Args:
            chunks: List of chunk dicts (output of Chunker.chunk_corpus).
            embeddings: List of embedding vectors, one per chunk.
                        Must have len(embeddings) == len(chunks).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings. "
                f"Each chunk must have exactly one embedding."
            )

        if len(chunks) == 0:
            raise ValueError("Cannot build an index from an empty chunk list.")

        vectors = np.array(embeddings).astype("float32")

        # Normalize to unit length so inner-product = cosine similarity
        faiss.normalize_L2(vectors)

        # IndexFlatIP: exact (non-approximate) inner-product search.
        # Fine for our research scale; revisit (e.g., IndexIVFFlat) only if
        # the index gets huge.
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks

    # ----- persistence -----

    def save(self, index_path: PathLike, chunks_path: PathLike) -> None:
        """Save the index and chunks to disk.

        Args:
            index_path: Path for the FAISS index file (typically *.faiss).
            chunks_path: Path for the chunks JSON file (typically *.json).
        """
        if self.index is None or self.chunks is None:
            raise RuntimeError(
                "Nothing to save: store is empty. Call build() first."
            )

        index_path = Path(index_path)
        chunks_path = Path(chunks_path)

        # Make sure parent directories exist
        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

    def load(self, index_path: PathLike, chunks_path: PathLike) -> None:
        """Load an index and chunks from disk.

        Args:
            index_path: Path to a FAISS index file previously written by save().
            chunks_path: Path to the matching chunks JSON file.
        """
        index_path = Path(index_path)
        chunks_path = Path(chunks_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        self.index = faiss.read_index(str(index_path))

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Inconsistent files: index has {self.index.ntotal} vectors "
                f"but chunks file has {len(self.chunks)} entries. The two "
                f"files may be from different runs."
            )

    # ----- search -----

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Find the top-k most similar chunks to a query embedding.

        Args:
            query_embedding: A single embedding vector (the query, embedded).
            k: Number of results to return.

        Returns:
            List of chunk dicts, each with an added "score" field (cosine
            similarity, between -1 and 1). Sorted from most to least similar.
        """
        if self.index is None or self.chunks is None:
            raise RuntimeError(
                "Cannot search: store is empty. Call build() or load() first."
            )

        if k > self.index.ntotal:
            k = self.index.ntotal  # don't ask for more than we have

        query_vec = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, k)

        results = []
        for i, idx in enumerate(indices[0]):
            chunk = dict(self.chunks[idx])  # shallow copy so we don't mutate
            chunk["score"] = float(scores[0][i])
            results.append(chunk)

        return results

    # ----- introspection -----

    @property
    def size(self) -> int:
        """Number of vectors in the index (0 if empty)."""
        return self.index.ntotal if self.index is not None else 0

    def __repr__(self) -> str:
        if self.index is None:
            return "VectorStore(empty)"
        return f"VectorStore(size={self.size})"