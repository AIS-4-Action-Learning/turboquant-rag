# Turboquant-RAG

`turboquant-rag` is the EPITA research project repository for validating
TurboQuant, a vector quantization algorithm, under a Retrieval-Augmented
Generation (RAG) environment.

The project evaluates whether TurboQuant can preserve the zero-shot accuracy
reported in the original paper while reducing the memory cost of Llama 3.1 KV
caches. The validation is performed against an uncompressed BF16 baseline in a
RAG setting where retrieved context and noisy context are used to study
hallucination behavior, retrieval robustness, and memory efficiency.

## Team

- Hamza El Hamdi
- Bernard Junior Seka
- Aishwarya Murthy

## Research Objective

We conduct a systems validation of TurboQuant integrated with Llama 3.1 8B in a
RAG pipeline. The core question is whether compressed KV-cache inference can
retain useful zero-shot behavior while lowering VRAM requirements and remaining
usable in retrieval-heavy generation workloads.

The research compares:

- `LlamaBF16`: baseline Llama 3.1 inference with an uncompressed BF16 KV cache.
- `LlamaCompressed`: Llama 3.1 inference with TurboQuant-compressed KV caches,
  including integer and mixed-precision bit widths such as 2.5-bit and 3.5-bit.

## Experiments

The project is organized around three distinct but related experiments.

### 1. TurboQuant in Llama 3.1 RAG

This experiment integrates TurboQuant into the Llama 3.1 RAG pipeline and
compares compressed-cache behavior against BF16. The target metrics are:

- Perplexity
- RMSE_key
- RMSE_value
- Zero-shot accuracy

Zero-shot accuracy is evaluated across three task families:

- Factual QA
- Cross-reference QA
- Out-of-scope QA

The goal is to validate the zero-shot accuracy claim from the original
TurboQuant paper in a RAG environment containing both retrieval-based context
and deliberately noisy context.

### 2. VRAM Profiling

This experiment measures the memory behavior of the compressed and BF16
systems. It tracks startup memory, peak memory, memory spikes, and latency under
different context lengths and cache strategies.

The current comparison notes are in [docs/COMPARISON.md](docs/COMPARISON.md).

### 3. RAG Benchmarking

This experiment benchmarks the RAG system against RAG-oriented evaluation work,
including MIRAGE and NoiserBench-style settings. The focus is on how retrieval,
noise, and out-of-scope context affect answer quality and hallucination
resistance when the generation backend uses either BF16 or TurboQuant-compressed
KV caches.

## Implementation Overview

The repository contains two connected implementation tracks:

- A modular RAG library with chunking, embedding, vector search, and swappable
  generators.
- A TurboQuant/Llama 3.1 integration that compresses the model KV cache during
  inference.

The RAG library allows the same retrieval pipeline to run with different
generation backends, including Gemini/OpenAI for prototyping and Llama 3.1 BF16
or TurboQuant for experiments.

The TurboQuant integration includes:

- CPU and CUDA TurboQuant bindings through `app/turboquant_simd.py` and
  `app/turboquant_simt.py`.
- Llama 3.1 model wrappers in `app/llama_models.py`.
- Modified Llama model code in `model/` for compressed KV-cache inference.
- Context initialization tooling in `scripts/initialize_context.py`.
- Notebook workflows for TurboQuant integration, profiling, and benchmarking.

## Repository Layout

```text
turboquant-rag/
|-- app/
|   |-- main.py                 # Example inference entry point
|   |-- llama_models.py         # BF16 and TurboQuant Llama wrappers
|   |-- turboquant_simd.py      # CPU TurboQuant bindings
|   `-- turboquant_simt.py      # CUDA TurboQuant bindings
|-- model/
|   |-- args.py                 # Modified Llama model arguments
|   `-- model.py                # Modified Transformer with compressed KV cache
|-- rag_library/
|   |-- chunker.py
|   |-- embedder.py
|   |-- generator.py
|   |-- rag.py
|   `-- vector_store.py
|-- scripts/
|   `-- initialize_context.py
|-- INSTALL                     # Native TurboQuant build script
`-- requirements.txt
```

## Documentation Guide

The `docs/` folder is the main source of project documentation:

- [docs/PROJECT_DESCRIPTION.md](docs/PROJECT_DESCRIPTION.md) explains the
  research motivation, architecture, TurboQuant variants, model integration,
  and runtime configuration.
- [docs/USER_MANUAL.md](docs/USER_MANUAL.md) gives the full reproducible
  workflow: environment setup, `.env` configuration, native build, context
  initialization, model loading, RAG assembly, and querying.
- [docs/CURRENT_IMPLEMENTATION.md](docs/CURRENT_IMPLEMENTATION.md) describes
  the current attention-layer behavior during prefill and autoregressive
  generation, including the compressed-cache bottlenecks.
- [docs/NEW_FEATURE.md](docs/NEW_FEATURE.md) documents recent correctness and
  performance changes, including direct GPU batch quantization, dense shadow
  cache decode optimization, and fused-kernel fixes.
- [docs/COMPARISON.md](docs/COMPARISON.md) records early VRAM and latency
  comparisons between dense-shadow and standard compressed-cache paths.
- [docs/ToDo.md](docs/ToDo.md) tracks experiment and metric implementation
  status.

Exported PDF versions of some documents are also present in `docs/`.

## Setup Summary

For the complete workflow, use [docs/USER_MANUAL.md](docs/USER_MANUAL.md). At a
high level:

1. Install the required toolchains:
   - Intel oneAPI `icx` for CPU/SIMD TurboQuant variants.
   - NVIDIA CUDA/NVCC for GPU/SIMT TurboQuant variants.
2. Install Python dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Create a `.env` file at the project root with Llama 3.1 8B checkpoint paths
   and TurboQuant runtime settings:

   ```ini
   PARAMS_PATH=/path/to/Llama3.1-8B/params.json
   CHECKPOINT_PATH=/path/to/Llama3.1-8B/consolidated.00.pth
   TOKENIZER_PATH=/path/to/Llama3.1-8B/tokenizer.model

   TURBOQUANT_VARIANT=simt-multi
   DEFAULT_BIT_WIDTH=3.5
   DEFAULT_DIMENSIONS=128
   DEFAULT_BLOCK_SIZE=8
   ```

4. Build the native TurboQuant libraries:

   ```bash
   source /opt/intel/oneapi/setvars.sh
   ./INSTALL
   ```

5. Optionally initialize a quantization context:

   ```bash
   python -m scripts.initialize_context
   ```

6. Run the application or notebooks for the relevant experiment:

   ```bash
   python -m app.main
   ```

## RAG Library Usage

The RAG pipeline is built from swappable components:

- `Chunker` splits documents into overlapping chunks.
- `Embedder` implementations produce retrieval vectors.
- `VectorStore` stores and searches embeddings with FAISS.
- `Generator` implementations answer questions from retrieved context.
- `RAG` orchestrates indexing and querying.

For prototyping, Gemini/OpenAI generators and embedders can be used. For the
research experiments, the important comparison is between `LlamaBF16` and
`LlamaCompressed` generation under the same retrieval and prompt conditions.

## Current Status

Implemented or documented:

- Modular RAG library.
- Gemini, OpenAI, and BGE embedding support.
- Gemini/OpenAI prototyping generators.
- BF16 and TurboQuant Llama generation wrappers.
- TurboQuant CPU/CUDA binding layer.
- Mixed-precision compressed KV-cache support.
- Direct GPU batch quantization path.
- Dense shadow cache decode optimization.
- Perplexity, MSE, and RMSE metric work.

Still in progress:

- Final zero-shot accuracy implementation for factual QA, cross-reference QA,
  and out-of-scope QA.
- Experiment runner that records metrics across bit widths, context
  configurations, and trials into CSV files.
- Full RAG benchmark execution against MIRAGE and NoiserBench-style settings.

## Compatibility

- Python 3.10+
- PyTorch 2.0+
- CUDA Toolkit 11+ or 12+ for GPU variants
- Intel oneAPI for CPU/SIMD variants
- Linux or Colab-style GPU environments are the primary supported targets

`DEFAULT_DIMENSIONS` must remain `128` for Llama 3.1 8B because it matches the
attention head dimension.
