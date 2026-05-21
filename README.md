# turboquant-rag

A modular RAG library built for the AIS4 TurboQuant compression research project at EPITA.

The library wraps a complete RAG pipeline (chunking, embedding, retrieval, generation) behind a single class with swappable components, so the team can plug in different LLM backends — Gemini and OpenAI for prototyping, BF16 or TurboQuant-compressed Llama 3.1 for benchmarks — without changing the rest of the code.

## Team

- Hamza El Hamdi
- Bernard Junior Seka
- Aishwarya Murthy

## Status

| Component | Status |
|---|---|
| Library structure (OOP refactor) | ✅ Done |
| Chunker, VectorStore, RAG orchestrator | ✅ Done |
| Gemini embedder / generator (free tier) | ✅ Done |
| OpenAI embedder / generator (paid) | ✅ Done |
| PDF extraction script | ✅ Done |
| BF16 Llama generator | ⏸ Stub (awaiting framework decision) |
| TurboQuant Llama generator | ⏸ Stub (awaiting Hamza's kernel integration) |

## Project layout

```
turboquant-rag/
├── README.md
├── requirements.txt
├── example.py            # End-to-end demo (Gemini)
├── extract.py            # PDF → corpus.json
├── data/                 # gitignored — populate locally
│   ├── pdfs/             # Source PDFs go here
│   ├── corpus_raw.json   # Output of extract.py — every page
│   ├── corpus.json       # Output of extract.py — filtered for RAG
│   └── extraction_report.md
└── rag_library/
    ├── __init__.py
    ├── chunker.py        # Chunker class
    ├── embedder.py       # Embedder ABC + OpenAI/Gemini implementations
    ├── generator.py      # Generator ABC + OpenAI/Gemini/Llama implementations
    ├── rag.py            # RAG orchestrator (the public face)
    └── vector_store.py   # FAISS index wrapper
```

---

## 1. Setup

### Clone the repo

```bash
git clone <repo-url>
cd turboquant-rag
git checkout develop
```

### Install dependencies

If you use Anaconda/miniconda (recommended on macOS):

```bash
python -m pip install -r requirements.txt
```

If you have multiple Python installs and want to be explicit:

```bash
python3 -m pip install -r requirements.txt
```

Verify installation:

```bash
python -c "import faiss, fitz; from google import genai; print('All dependencies installed')"
```

### Get a Gemini API key (free, no credit card)

1. Go to https://aistudio.google.com/apikey
2. Sign in with any Google account
3. Click **Create API key**
4. Copy the key

### Configure your API key locally

Create a `.env` file at the project root:

```bash
echo "GEMINI_API_KEY=your-key-here" > .env
```

`.env` is gitignored — your key never gets pushed.

> **Note**: Gemini's free tier uses your prompts to improve their models. For our research (a public textbook), this is fine. Don't send anything sensitive.

### (Optional) OpenAI key

Only needed if you want to use `OpenAIGenerator` / `OpenAIEmbedder` instead of the Gemini defaults. OpenAI is paid (no real free tier).

```bash
echo "OPENAI_API_KEY=sk-..." >> .env
```

Set a hard spending limit at https://platform.openai.com/settings/organization/limits before adding any credit.

---

## 2. Get the source corpus

The PDF and processed corpora are not in the repo (gitignored). Each teammate downloads the PDF locally and runs the extraction script.

### Download the textbook

Our research corpus is *Dive into Deep Learning* (free, open-source):

1. Download from https://d2l.ai/ (the PDF is freely available)
2. Place it at `data/pdfs/d2l-en.pdf`:

```bash
mkdir -p data/pdfs
mv ~/Downloads/d2l-en.pdf data/pdfs/
```

### Run the extraction script

```bash
python extract.py
```

This takes 1–2 minutes for the full d2l book. It produces three files in `data/`:

- `corpus_raw.json` — every page, no filtering (a checkpoint)
- `corpus.json` — filtered version, ready for RAG indexing
- `extraction_report.md` — per-page summary for inspecting what got kept/dropped

Verify the result:

```bash
python -c "
import json
with open('data/corpus.json') as f:
    corpus = json.load(f)
print(f'Pages: {len(corpus)}')
print(f'First: p.{corpus[0][\"page\"]}')
print(f'Last: p.{corpus[-1][\"page\"]}')
"
```

Expected output: ~1068 pages, first p.50, last p.1128 (front matter and references filtered out).

If your numbers are wildly different, you may have a different PDF version. Open `data/extraction_report.md` and adjust `SKIP_RULES` at the top of `extract.py`.

---

## 3. Run the end-to-end demo

```bash
python example.py
```

This script:

1. Loads your Gemini API key from `.env`
2. Builds a small in-memory corpus (3 pages of deep learning content)
3. Indexes it with `GeminiEmbedder` + FAISS
4. Saves the index to `data/demo_index.faiss` and `data/demo_chunks.json`
5. Loads it back into a fresh `RAG` instance
6. Asks "How does backpropagation compute gradients?" and prints the answer + retrieved chunks

Expected runtime: ~30 seconds (most of it is rate-limit throttling, which is the price of the free tier).

---

## 4. Use the library

The library exposes a single public class — `RAG` — with swappable components.

### Prototyping with Gemini (recommended — free)

```python
import os
from dotenv import load_dotenv
from google import genai

from rag_library import RAG, GeminiEmbedder, GeminiGenerator

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

rag = RAG(
    embedder=GeminiEmbedder(client),
    generator=GeminiGenerator(client),
)

# One-time setup
rag.build_index("data/corpus.json")
rag.save("data/index.faiss", "data/chunks.json")

# Later, after restart:
rag.load("data/index.faiss", "data/chunks.json")
result = rag.query("What is backpropagation?")
print(result["answer"])
```

### Prototyping with OpenAI (paid alternative)

```python
from openai import OpenAI
from rag_library import RAG, OpenAIEmbedder, OpenAIGenerator

client = OpenAI(api_key="sk-...")
rag = RAG(
    embedder=OpenAIEmbedder(client),
    generator=OpenAIGenerator(client),
)
# ... same workflow as Gemini
```

### Llama generators (BF16 or TurboQuant)

Llama-based generators can still be used for answering, but retrieval should
use a pretrained embedder (Gemini or OpenAI). Build the index with a
pretrained embedder and swap only the generator.

In every case, the rest of the pipeline (`build_index`, `query`, `save`, `load`) is identical.

---

## 5. Troubleshooting

**`ModuleNotFoundError: No module named 'fitz'` (or similar)**

Your `pip` and `python` are pointing to different installations. Use `python -m pip install ...` instead of just `pip install ...`.

**`GEMINI_API_KEY not found`**

Check that `.env` exists at the project root and contains `GEMINI_API_KEY=your-key`. If running from a different folder, make sure the working directory is the project root.

**`429 Resource Exhausted` from Gemini**

You hit the free-tier rate limit. The library throttles automatically, but if you run multiple scripts in parallel, you can still exceed the limit. Wait a minute and retry, or switch to a higher-RPM model:
```python
GeminiGenerator(client, model="gemini-2.5-flash")  # 10 RPM instead of 15
```

**Extraction script gives wrong page counts**

Your PDF may be a different version. Open `data/extraction_report.md`, find the page boundaries for front matter and bibliography, and update `SKIP_RULES` in `extract.py`.

**`TypeError: unsupported operand type(s) for |`**

You're on Python 3.9 and something is using `str | None` syntax. The library is fully 3.9-compatible; this would mean a regression in a recent edit. Paste the error in the team channel.

---

## 6. Compatibility

All library code is Python 3.9+ compatible, verified by AST parsing with `feature_version=(3, 9)`. Tested locally on macOS, Kaggle Notebooks, and Google Colab.

## 7. Branching workflow

- `main` — release-stable
- `develop` — integration branch (everyone PRs into this one)
- `feat/<name>` — feature branches for your work

Always branch off `develop`, never `main`. PR back into `develop`.