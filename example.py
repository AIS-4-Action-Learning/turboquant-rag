"""
example.py — minimal end-to-end demo of the rag_library, using Gemini.

Run this from the project root (the folder containing rag_library/):

    python example.py

What it does:
  1. Creates a small in-memory corpus (no PDF parsing here — see extract.py).
  2. Builds a RAG pipeline using Gemini for embeddings and generation.
  3. Indexes the corpus.
  4. Saves the index to disk.
  5. Loads it back into a fresh RAG instance.
  6. Runs a query and prints the result.

Requirements:
  - GEMINI_API_KEY in your environment (or in a .env file at the project root).
    Get one for free at https://aistudio.google.com/apikey — no credit card required.
  - pip install google-genai faiss-cpu numpy python-dotenv
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from rag_library import RAG, GeminiEmbedder, GeminiGenerator

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

load_dotenv()  # reads .env if present

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Set it in your environment or in a .env file.\n"
        "Get a free key (no credit card) at https://aistudio.google.com/apikey"
    )

client = genai.Client(api_key=api_key)

# Where to save the index. Adjust to taste.
DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "demo_index.faiss"
CHUNKS_PATH = DATA_DIR / "demo_chunks.json"


# ---------------------------------------------------------------------------
# 2. Define a tiny corpus (replace with your real corpus)
# ---------------------------------------------------------------------------

# In your real workflow, you'd load this from data/corpus.json — produced by
# extract.py. Same shape: a list of page dicts with "text", "source", "page".
demo_corpus = [
    {
        "source": "deep_learning.pdf",
        "page": 1,
        "text": (
            "Backpropagation is a fundamental algorithm in deep learning. "
            "It computes gradients of the loss function with respect to "
            "network weights using the chain rule, propagating errors "
            "backward from the output layer to the input layer. "
        ) * 5,
    },
    {
        "source": "deep_learning.pdf",
        "page": 2,
        "text": (
            "Convolutional neural networks (CNNs) are designed for "
            "processing grid-like data such as images. They use shared "
            "weights and local receptive fields to achieve translation "
            "invariance and parameter efficiency. "
        ) * 5,
    },
    {
        "source": "deep_learning.pdf",
        "page": 3,
        "text": (
            "Attention mechanisms allow neural networks to focus on "
            "relevant parts of the input. Self-attention, the core of "
            "the Transformer architecture, computes weighted sums over "
            "all positions in the sequence. "
        ) * 5,
    },
]


# ---------------------------------------------------------------------------
# 3. Build a RAG pipeline
# ---------------------------------------------------------------------------

# This is where swappability pays off. For research benchmarks you'd replace
# GeminiEmbedder/GeminiGenerator with the Llama versions. Everything else stays.
rag = RAG(
    embedder=GeminiEmbedder(client),
    generator=GeminiGenerator(client),
    top_k=3,
)
print(f"Created: {rag!r}")


# ---------------------------------------------------------------------------
# 4. Index the corpus
# ---------------------------------------------------------------------------

# NOTE: GeminiEmbedder throttles requests to stay under the free-tier rate
# limit, so indexing takes a few seconds per batch. For our 3-page demo this
# is essentially instant; for the full d2l corpus it'll take several minutes.
print("\nBuilding index (may take a moment due to free-tier rate limits)...")
rag.build_index(demo_corpus)
print(f"After indexing: {rag!r}")


# ---------------------------------------------------------------------------
# 5. Save and reload (just to demonstrate persistence works)
# ---------------------------------------------------------------------------

print(f"\nSaving to {INDEX_PATH} and {CHUNKS_PATH}...")
rag.save(INDEX_PATH, CHUNKS_PATH)

print("Loading into a fresh RAG instance...")
rag2 = RAG(
    embedder=GeminiEmbedder(client),
    generator=GeminiGenerator(client),
    top_k=3,
)
rag2.load(INDEX_PATH, CHUNKS_PATH)
print(f"Loaded: {rag2!r}")


# ---------------------------------------------------------------------------
# 6. Query
# ---------------------------------------------------------------------------

question = "How does backpropagation compute gradients?"
print(f"\nQuestion: {question}")

result = rag2.query(question)

print("\nAnswer:")
print(result["answer"])

print("\nRetrieved chunks:")
for chunk in result["retrieved"]:
    print(
        f"  - {chunk['source']} p.{chunk['page']} "
        f"(score: {chunk['score']:.3f})"
    )