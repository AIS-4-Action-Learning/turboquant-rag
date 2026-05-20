"""
rag_library — A modular RAG pipeline for the TurboQuant research benchmarks.

Public API:
    RAG: The main orchestrator class.
    Chunker: Splits a corpus into chunks.
    VectorStore: FAISS-backed similarity search.

    Generators (swappable):
        Generator                — abstract base class
        OpenAIGenerator          — OpenAI chat completion (paid)
        GeminiGenerator          — Google Gemini (free tier, recommended for prototyping)
        BF16LlamaGenerator       — Llama 3.1 8B BF16 (research baseline) [stub]
        TurboQuantLlamaGenerator — Llama 3.1 8B + TurboQuant (research) [stub]

    Embedders (swappable):
        Embedder                — abstract base class
        OpenAIEmbedder          — OpenAI text-embedding-3-small (paid)
        GeminiEmbedder          — Gemini embedding-001 (free tier, recommended)
        BF16LlamaEmbedder       — Llama 3.1 8B BF16 hidden states [stub]
        TurboQuantLlamaEmbedder — Llama 3.1 8B + TurboQuant hidden states [stub]

Example usage:

    # Recommended for prototyping: Gemini (free, no credit card)
    >>> from google import genai
    >>> from rag_library import RAG, GeminiEmbedder, GeminiGenerator
    >>> client = genai.Client(api_key="...")  # or set GEMINI_API_KEY env var
    >>> rag = RAG(
    ...     embedder=GeminiEmbedder(client),
    ...     generator=GeminiGenerator(client),
    ... )
    >>> rag.build_index("data/corpus.json")
    >>> result = rag.query("What is backpropagation?")
    >>> print(result["answer"])

    # Research benchmark — BF16 baseline
    >>> from rag_library import RAG, BF16LlamaGenerator, BF16LlamaEmbedder
    >>> rag = RAG(
    ...     embedder=BF16LlamaEmbedder(model_path="meta-llama/Llama-3.1-8B"),
    ...     generator=BF16LlamaGenerator(model_path="meta-llama/Llama-3.1-8B"),
    ... )

    # Research benchmark — TurboQuant compressed
    >>> from rag_library import RAG, TurboQuantLlamaGenerator, TurboQuantLlamaEmbedder
    >>> rag = RAG(
    ...     embedder=TurboQuantLlamaEmbedder(model_path="...", bit_width=3),
    ...     generator=TurboQuantLlamaGenerator(model_path="...", bit_width=3),
    ... )
"""

from .chunker import Chunker
from .embedder import (
    Embedder,
    OpenAIEmbedder,
    GeminiEmbedder,
    Llama3Embedder,
)
from .generator import (
    Generator,
    OpenAIGenerator,
    GeminiGenerator,
    BF16LlamaGenerator,
    TurboQuantLlamaGenerator,
)
from .rag import RAG
from .vector_store import VectorStore

__all__ = [
    # Main entry point
    "RAG",
    # Chunking
    "Chunker",
    # Vector store
    "VectorStore",
    # Generators
    "Generator",
    "OpenAIGenerator",
    "GeminiGenerator",
    "BF16LlamaGenerator",
    "TurboQuantLlamaGenerator",
    # Embedders
    "Embedder",
    "OpenAIEmbedder",
    "GeminiEmbedder",
    "Llama3Embedder",
]
