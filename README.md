# turboquant-rag

# rag-library

A modular RAG library built for the AIS4 TurboQuant compression research project at EPITA.

The library wraps a complete RAG pipeline (chunking, embedding, retrieval, generation) behind a single class with swappable components, so the team can plug in OpenAI for prototyping or compressed/uncompressed Llama 3.1 for benchmarks without changing the rest of the code.

## Team

- Hamza El Hamdi
- Bernard Junior Seka
- Aishwarya Murthy

## Status

| Component | Status |
|---|---|
| Library structure (OOP refactor) |  Done |
| Chunker, VectorStore, RAG orchestrator |  Done |
| OpenAI embedder / generator |  Done |
| PDF extraction script |  Done |
| BF16 Llama generator / embedder | ⏸ Stub (awaiting framework decision) |
| TurboQuant Llama generator / embedder | ⏸ Stub (awaiting  kernel integration) |

## Project layout

```
turboquant-rag/
├── README.md
├── requirements.txt
├── example.py            # End-to-end demo
├── extract.py            # PDF → corpus.json
├── data/                 # (gitignored — populate locally)
│   ├── pdfs/             # Put source PDFs here
│   ├── corpus_raw.json   # Output: every page
│   ├── corpus.json       # Output: filtered for RAG
│   └── extraction_report.md
└── rag_library/
    ├── __init__.py
    ├── chunker.py        # Chunker class
    ├── embedder.py       # Embedder ABC + OpenAI/Llama implementations
    ├── generator.py      # Generator ABC + OpenAI/Llama implementations
    ├── rag.py            # RAG orchestrator (the public face)
    └── vector_store.py   # FAISS index wrapper
```

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd turboquant-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key in a .env file (or environment variable)
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Put your source PDF in data/pdfs/
mkdir -p data/pdfs
cp /path/to/your/book.pdf data/pdfs/
```

## Extracting a corpus

```bash
# First pass: extract everything, then look at the report to decide what to skip
python extract.py

# Look at data/extraction_report.md to identify front matter, references, etc.
# Update SKIP_RULES at the top of extract.py with the page ranges to drop.

# Re-run with skip rules applied
python extract.py
```

Output: `data/corpus.json` is ready for `RAG.build_index()`.

## Using the library

```python
from openai import OpenAI
from rag_library import RAG, OpenAIEmbedder, OpenAIGenerator

client = OpenAI()  # reads OPENAI_API_KEY from env

rag = RAG(
    embedder=OpenAIEmbedder(client),
    generator=OpenAIGenerator(client),
)
rag.build_index("data/corpus.json")
rag.save("data/index.faiss", "data/chunks.json")

# Later, after restart:
rag.load("data/index.faiss", "data/chunks.json")
result = rag.query("What is backpropagation?")
print(result["answer"])
```

For the research benchmarks, swap the generator/embedder for the Llama versions (once implemented):

```python
from rag_library import RAG, BF16LlamaGenerator, BF16LlamaEmbedder
# or:
from rag_library import RAG, TurboQuantLlamaGenerator, TurboQuantLlamaEmbedder
```

Everything else stays identical.

## Quick demo

`example.py` runs the full pipeline on a small in-memory corpus. Useful for sanity-checking that your environment is set up correctly:

```bash
python example.py
```
