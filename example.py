"""
example.py — minimal end-to-end demo of the rag_library.

Run this from the project root (the folder containing rag_library/):

    python3 example.py

What it does:
  1. Creates a small in-memory corpus (no PDF parsing here — see your
     extraction script for that).
  2. Builds a RAG pipeline using OpenAI for embeddings and generation.
  3. Indexes the corpus.
  4. Saves the index to disk.
  5. Loads it back into a fresh RAG instance.
  6. Runs a query and prints the result.

Requirements:
  - OPENAI_API_KEY in your environment (or in a .env file at the project root).
  - pip install openai faiss-cpu numpy python-dotenv
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag_library import RAG, OpenAIEmbedder, OpenAIGenerator

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

load_dotenv()  # reads .env if present

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Set it in your environment or in a .env file."
    )

client = OpenAI(api_key=api_key)

# Where to save the index. Adjust to taste.
DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "corpus.json"
CHUNKS_PATH = DATA_DIR / "corpus.json"


# ---------------------------------------------------------------------------
# 2. Define a tiny corpus (replace with your real corpus)
# ---------------------------------------------------------------------------

# In your real workflow, you'd load this from data/corpus.json — produced by
# your PDF extraction script. The shape is the same: a list of page dicts
# each with "text", "source", and "page" keys.
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

# This is where the swappability pays off. For research benchmarks you'd
# replace OpenAIEmbedder/OpenAIGenerator with the Llama versions. The rest
# of the script stays the same.
rag = RAG(
    embedder=OpenAIEmbedder(client),
    generator=OpenAIGenerator(client),
    top_k=3,
)
print(f"Created: {rag!r}")


# ---------------------------------------------------------------------------
# 4. Index the corpus
# ---------------------------------------------------------------------------

print("\nBuilding index...")
rag.build_index(demo_corpus)
print(f"After indexing: {rag!r}")


# ---------------------------------------------------------------------------
# 5. Save and reload (just to demonstrate persistence works)
# ---------------------------------------------------------------------------

print(f"\nSaving to {INDEX_PATH} and {CHUNKS_PATH}...")
rag.save(INDEX_PATH, CHUNKS_PATH)

print("Loading into a fresh RAG instance...")
rag2 = RAG(
    embedder=OpenAIEmbedder(client),
    generator=OpenAIGenerator(client),
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