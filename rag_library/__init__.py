"""
rag_library — A modular RAG pipeline for the TurboQuant research benchmarks.

The package exposes a lazy public API so importing one submodule does not
force the whole stack to load. That keeps optional dependencies isolated:
embedding-only paths do not need the Llama stack, and generator-only paths do
not need embedding backends until they are actually used.
"""

from importlib import import_module

__all__ = [
    "RAG",
    "Chunker",
    "VectorStore",
    "Generator",
    "OpenAIGenerator",
    "GeminiGenerator",
    "BF16LlamaGenerator",
    "TurboQuantLlamaGenerator",
    "Embedder",
    "OpenAIEmbedder",
    "GeminiEmbedder",
    "BGEmbedder",
]

_EXPORTS = {
    "RAG": ("rag_library.rag", "RAG"),
    "Chunker": ("rag_library.chunker", "Chunker"),
    "VectorStore": ("rag_library.vector_store", "VectorStore"),
    "Generator": ("rag_library.generator", "Generator"),
    "OpenAIGenerator": ("rag_library.generator", "OpenAIGenerator"),
    "GeminiGenerator": ("rag_library.generator", "GeminiGenerator"),
    "BF16LlamaGenerator": ("rag_library.generator", "BF16LlamaGenerator"),
    "TurboQuantLlamaGenerator": ("rag_library.generator", "TurboQuantLlamaGenerator"),
    "Embedder": ("rag_library.embedder", "Embedder"),
    "OpenAIEmbedder": ("rag_library.embedder", "OpenAIEmbedder"),
    "GeminiEmbedder": ("rag_library.embedder", "GeminiEmbedder"),
    "BGEmbedder": ("rag_library.embedder", "BGEmbedder"),
}


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
