"""
Embedder interface and implementations.

An Embedder turns text into vectors. Like Generator, this is swappable so
we can compare OpenAI and Gemini embeddings in our research.

Class hierarchy:
    Embedder (ABC)
    ├── OpenAIEmbedder
    └── GeminiEmbedder
"""

import time
from abc import ABC, abstractmethod
from typing import List, cast
import numpy as np

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
# Gemini implementation (Google's free-tier-friendly API)
# ---------------------------------------------------------------------------

class GeminiEmbedder(Embedder):
    """Embedder that uses Google's Gemini embedding API.

    Uses `gemini-embedding-001` by default — the production embedding model.
    Like GeminiGenerator, this is free-tier-friendly and requires no payment
    method.

    Handles batching and rate-limiting internally so that indexing a large
    corpus doesn't hit the per-minute request cap.
    """

    # Gemini's embed_content can take a list of inputs in one call, which
    # counts as one request against the rate limit — much better than embedding
    # one chunk at a time. Conservative batch size keeps payloads small.
    DEFAULT_BATCH_SIZE = 100

    # Embedding endpoint shares the per-project RPM limit. Sleep between batches
    # to stay safely under 15 RPM on the free tier.
    DEFAULT_MIN_INTERVAL_SECONDS = 4.5

    def __init__(
        self,
        client,
        model: str = "gemini-embedding-001",
        batch_size: int = DEFAULT_BATCH_SIZE,
        output_dimensionality: int = 768,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
    ):
        """
        Args:
            client: An initialized google.genai.Client.
            model: Gemini embedding model name. Default is the production model.
            batch_size: Number of texts per embed_content call.
            output_dimensionality: Embedding vector length. gemini-embedding-001
                                   supports 768, 1536, or 3072 (default 3072).
                                   We default to 768 to keep the FAISS index
                                   small and retrieval fast.
            min_interval_seconds: Minimum seconds between batched calls.
        """
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.output_dimensionality = output_dimensionality
        self.min_interval_seconds = min_interval_seconds
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        """Sleep if needed to respect the per-minute rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_time = time.time()

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Lazy import so the SDK is only required when actually used.
        from google.genai import types

        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            self._throttle()

            response = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.output_dimensionality,
                ),
            )

            embeddings.extend([e.values for e in response.embeddings])

        return embeddings


class BGEmbedder(Embedder):
    def __init__(self, batch_size: int = 12):
        # Import lazily so the module can be imported without requiring the
        # full FlagEmbedding/transformers stack unless this embedder is used.
        from FlagEmbedding import FlagModel

        self.model = FlagModel(
            'BAAI/bge-small-en-v1.5',
            use_fp16=False,
            device='cpu'
        )

        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            dense_embeddings = self.model.encode(
                texts,
                self.batch_size,
            )


            return dense_embeddings.tolist()
        except Exception as e:
            print(e)
            return [[]]
