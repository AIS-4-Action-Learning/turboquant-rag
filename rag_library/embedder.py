"""
Embedder interface and implementations.

An Embedder turns text into vectors. Like Generator, this is swappable so
we can compare OpenAI embeddings vs Llama-based embeddings in our research.

Class hierarchy:
    Embedder (ABC)
    ├── OpenAIEmbedder
    ├── BF16LlamaEmbedder       ──┐
    └── TurboQuantLlamaEmbedder  ──┴── both inherit shared logic from _LlamaEmbedderBase
"""

from abc import ABC, abstractmethod
from typing import List


# ---------------------------------------------------------------------------
# Public abstract base class
# ---------------------------------------------------------------------------

class Embedder(ABC):
    """Abstract base class for embedders.

    Any concrete embedder must inherit from this class and implement
    the `embed` method.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors (each a list of floats).
            len(returned) == len(texts).
        """
        pass


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIEmbedder(Embedder):
    """Embedder that uses OpenAI's embeddings API.

    Handles batching internally to avoid hitting API limits on large corpora.
    """

    def __init__(
        self,
        client,
        model: str = "text-embedding-3-small",
        batch_size: int = 50,
        cost_tracker=None,
    ):
        """
        Args:
            client: An initialized OpenAI client (passed in, not created here).
            model: The OpenAI embedding model name.
            batch_size: Number of texts to embed per API call.
            cost_tracker: Optional callable that takes (response, is_embedding=True)
                          to track API costs. Pass None to disable.
        """
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.cost_tracker = cost_tracker

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)

            if self.cost_tracker is not None:
                self.cost_tracker(response, is_embedding=True)

            embeddings.extend([d.embedding for d in response.data])

        return embeddings


# ---------------------------------------------------------------------------
# Shared base for Llama-based embedders
# ---------------------------------------------------------------------------

class _LlamaEmbedderBase:
    """Shared logic for Llama-based embedders (BF16 and TurboQuant).

    Holds tokenization, batching, and pooling logic that doesn't depend on
    the inference path. Concrete subclasses below differ in how they
    actually run the forward pass.

    NOTE: Llama is a generative model; using it as an embedder typically
    means taking hidden states from a chosen layer and pooling (mean,
    last-token, etc.). The exact strategy will be decided when implementing.
    """

    DEFAULT_BATCH_SIZE = 8  # smaller than OpenAI's 50; local GPU memory limit
    DEFAULT_POOLING = "mean"  # mean | last_token | cls — to decide on impl

    def _batch(self, texts: List[str], batch_size: int):
        """Yield batches of texts for embedding."""
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]


# ---------------------------------------------------------------------------
# Llama BF16 (baseline) — STUB, to be implemented
# ---------------------------------------------------------------------------

class BF16LlamaEmbedder(_LlamaEmbedderBase, Embedder):
    """Embedder using BF16 (uncompressed) Llama 3.1 8B.

    BASELINE for our research benchmarks. Llama 3.1 is used as both
    generator and embedder so the entire RAG pipeline runs on a single
    model — this matches our research narrative of evaluating TurboQuant
    in a self-contained Llama-based pipeline.

    STATUS: stub. Awaiting framework decision and pooling strategy.
    """

    def __init__(
        self,
        model_path: str,
        batch_size: int = _LlamaEmbedderBase.DEFAULT_BATCH_SIZE,
        pooling: str = _LlamaEmbedderBase.DEFAULT_POOLING,
    ):
        """
        Args:
            model_path: Path or HF identifier for Llama 3.1 8B BF16.
            batch_size: Number of texts per forward pass.
            pooling: How to pool hidden states into a single vector
                     ("mean", "last_token", or "cls").
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.pooling = pooling

        # TODO: load the model here once framework is decided.

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "BF16LlamaEmbedder.embed not yet implemented. "
            "Awaiting framework decision and integration."
        )


# ---------------------------------------------------------------------------
# Llama TurboQuant (compressed) — STUB, to be implemented
# ---------------------------------------------------------------------------

class TurboQuantLlamaEmbedder(_LlamaEmbedderBase, Embedder):
    """Embedder using TurboQuant-compressed Llama 3.1 8B.

    EXPERIMENTAL configuration. Same Llama as BF16LlamaEmbedder but with
    TurboQuant compression applied. Used to evaluate whether compression
    affects the quality of retrieval embeddings, not just generation.

    STATUS: stub. Awaiting Hamza's TurboQuant kernel integration.
    """

    def __init__(
        self,
        model_path: str,
        bit_width: int = 3,
        batch_size: int = _LlamaEmbedderBase.DEFAULT_BATCH_SIZE,
        pooling: str = _LlamaEmbedderBase.DEFAULT_POOLING,
    ):
        """
        Args:
            model_path: Path or HF identifier for Llama 3.1 8B.
            bit_width: TurboQuant compression bit-width (2, 3, or 4).
            batch_size: Number of texts per forward pass.
            pooling: How to pool hidden states into a single vector.
        """
        self.model_path = model_path
        self.bit_width = bit_width
        self.batch_size = batch_size
        self.pooling = pooling

        # TODO: load model + apply TurboQuant compression once available.

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "TurboQuantLlamaEmbedder.embed not yet implemented. "
            "Awaiting Hamza's TurboQuant kernel integration."
        )