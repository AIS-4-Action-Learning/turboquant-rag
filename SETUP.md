# TurboQuant RAG Setup Guide

This guide covers complete setup and configuration for the TurboQuant RAG project, including building variants, configuring the environment, initializing contexts, and running inference.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure Overview](#project-structure-overview)
3. [Building TurboQuant Variants](#building-turboquant-variants)
4. [Environment Configuration](#environment-configuration)
5. [Initializing the Context](#initializing-the-context)
6. [Running Inference](#running-inference)
7. [Configuration Reference](#configuration-reference)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Operating System**: Linux (tested on Fedora)
- **Compiler**: GCC with C11 and OpenMP support
- **CUDA** (optional): For GPU acceleration (simt variants)
- **Python**: 3.10+
- **Virtual Environment**: Recommended (venv, virtualenv, or uv)

### Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Core dependencies include:
# - torch (PyTorch)
# - fairscale (for model parallelism)
# - python-dotenv (environment configuration)
# - tiktoken (tokenizer support)
```

---

## Project Structure Overview

```
turboquant-rag/
├── .env                      # Environment configuration (created by you)
├── app/
│   ├── __init__.py          # App initialization, env loading, library path resolution
│   ├── turboquant_simd.py   # CPU (SIMD) implementation
│   ├── turboquant_simt.py   # CUDA (SIMT) implementation
│   ├── llama_models.py      # Llama model wrappers
│   └── main.py              # Example usage script
├── scripts/
│   └── initialize_context.py # Context initialization script
├── artifacts/                # Built TurboQuant libraries and contexts
│   ├── simd/                # CPU single-threaded
│   ├── simd-multi/          # CPU multi-threaded
│   ├── simt/                # CUDA single-stream
│   ├── simt-multi/          # CUDA multi-stream
│   └── turboquant_ctx_*.bin # Generated context files
├── model/                    # Llama model files
│   ├── params.json          # Model hyperparameters
│   ├── tokenizer.model      # Tokenizer
│   └── consolidated.00.pth  # Model weights
└── INSTALL.sh               # Build script for TurboQuant variants
```

---

## Building TurboQuant Variants

TurboQuant provides multiple implementations optimized for different hardware:

| Variant | Description | Hardware |
|---------|-------------|----------|
| `simd` | CPU single-threaded | Any CPU |
| `simd-multi` | CPU multi-threaded/batch | Multi-core CPU |
| `simt` | CUDA single-stream | NVIDIA GPU |
| `simt-multi` | CUDA multi-stream/batch | NVIDIA GPU |

### Running the Build

```bash
# Build all variants
./INSTALL.sh

# Verify builds
ls artifacts/
# Should show: simd/ simd-multi/ simt/ simt-multi/

# Check library exists
ls artifacts/simd/libturboquant.so
ls artifacts/simd-multi/libturboquant.so
```

The `INSTALL.sh` script:
1. Builds CPU variants using GCC with OpenMP
2. Builds CUDA variants using NVCC (if CUDA is available)
3. Places libraries in `artifacts/<variant>/libturboquant.so`

---

## Environment Configuration

Configuration is managed through environment variables, typically set in a `.env` file in the project root.

### Creating the .env File

Create `.env` in the project root directory:

```bash
touch .env
```

### Complete .env Template

```ini
# =============================================================================
# Model Paths
# =============================================================================
# Paths to Llama model files (relative to project root)
PARAMS_PATH=./model/params.json
CHECKPOINT_PATH=./model/consolidated.00.pth
TOKENIZER_PATH=./model/tokenizer.model

# =============================================================================
# TurboQuant Library Configuration
# =============================================================================
# Priority 1: Direct library path (overrides variant selection)
# Use this for explicit control over which library to load
LIB_PATH=./artifacts/simd/libturboquant.so

# Priority 2: Context file path (set automatically by initialize_context.py)
# Leave empty to auto-generate, or set to use existing context
CONTEXT_PATH=

# Priority 3: TurboQuant source root (for building)
TURBOQUANT_ROOT=../TurboQuantQuantization

# =============================================================================
# TurboQuant Variant Selection
# =============================================================================
# Controls which implementation variant to use when LIB_PATH is not set
# Valid values: auto, simd, simd-multi, simt, simt-multi
# Also accepts aliases: cpu, cpu-batch, cpu-multi, cuda, cuda-batch

# auto: Select based on device (cpu=simd/simd-multi, cuda=simt/simt-multi)
#       and batch mode
TURBOQUANT_VARIANT=auto

# Explicit examples:
# TURBOQUANT_VARIANT=simd          # CPU single-threaded
# TURBOQUANT_VARIANT=simd-multi    # CPU multi-threaded (batch)
# TURBOQUANT_VARIANT=simt          # CUDA single-stream
# TURBOQUANT_VARIANT=simt-multi    # CUDA multi-stream (batch)

# =============================================================================
# TurboQuant Quantization Parameters
# =============================================================================
# Bit width for quantization (Algorithm 2 uses bit_width-1 for MSE,
# 1 bit reserved for QJL residual sign)
# Valid: 2-8, Recommended: 3-4
DEFAULT_BIT_WIDTH=3

# Block/dimension size for quantization (vector dimension quantized together)
# Must match model head dimension or be a divisor
# For Llama 3.1 8B: head_dim=128
DEFAULT_DIMENSIONS=128

# Number of streams/threads for batch processing (batch parallelism level)
# For simd-multi: thread pool size
# For simt-multi: CUDA stream count
DEFAULT_BLOCK_SIZE=8
```

### Configuration Priority

The system uses the following priority for configuration:

1. **LIB_PATH** (highest priority): If set, loads this exact library
2. **TURBOQUANT_VARIANT**: Selects variant directory under `artifacts/`
3. **Auto-detection** (lowest): Based on `device` and `is_batch` parameters

### Example Configurations

#### CPU-Only Setup (No CUDA)

```ini
LIB_PATH=./artifacts/simd/libturboquant.so
TURBOQUANT_VARIANT=simd
DEFAULT_BIT_WIDTH=3
DEFAULT_DIMENSIONS=128
DEFAULT_BLOCK_SIZE=8
```

#### CPU Batch Processing (Multi-threaded)

```ini
LIB_PATH=./artifacts/simd-multi/libturboquant.so
TURBOQUANT_VARIANT=simd-multi
DEFAULT_BIT_WIDTH=3
DEFAULT_DIMENSIONS=128
DEFAULT_BLOCK_SIZE=16
```

#### CUDA Single-Stream

```ini
LIB_PATH=./artifacts/simt/libturboquant.so
TURBOQUANT_VARIANT=simt
DEFAULT_BIT_WIDTH=3
DEFAULT_DIMENSIONS=128
DEFAULT_BLOCK_SIZE=8
```

#### CUDA Multi-Stream (Batch)

```ini
LIB_PATH=./artifacts/simt-multi/libturboquant.so
TURBOQUANT_VARIANT=simt-multi
DEFAULT_BIT_WIDTH=3
DEFAULT_DIMENSIONS=128
DEFAULT_BLOCK_SIZE=8
```

#### Auto-Select with Explicit Context

```ini
# Let system auto-select based on device
TURBOQUANT_VARIANT=auto

# But use specific pre-generated context
CONTEXT_PATH=./artifacts/turboquant_ctx_simd_128d_3b.bin
```

---

## Initializing the Context

The TurboQuant library requires an initialized context containing Haar rotation matrices, codebooks, and other quantization parameters. This is generated based on your bit-width and dimension settings.

### When to Run Initialization

Run `initialize_context.py` when:
- First time setup
- Changing `DEFAULT_BIT_WIDTH`
- Changing `DEFAULT_DIMENSIONS`
- Changing `TURBOQUANT_VARIANT` (different library)
- Context file is missing or corrupted

### Running Initialization

```bash
# Ensure your .env is configured
# Recommended: set LIB_PATH to specific variant

# Run initialization
python -m scripts.initialize_context
```

### What Initialization Does

1. **Reads .env configuration**: Loads bit-width, dimensions, block size, variant
2. **Locates library**: Uses LIB_PATH or finds based on variant
3. **Detects API type**: 
   - SIMD variants use global state API (`turboquant_init`, `turboquant_clean`)
   - SIMT variants use context pointer API (`turboquant_init`, `turboquant_context_destroy`)
4. **Initializes context**: Calls appropriate C API functions
5. **Saves context**: Writes binary file to `artifacts/turboquant_ctx_<variant>_<dims>d_<bits>b.bin`
6. **Updates .env**: Sets `CONTEXT_PATH` to the generated file

### Initialization Output Example

```
============================================================
TurboQuant Context Initialization
============================================================

Configuration:
  Variant: simd
  Bit Width: 3
  Dimensions: 128
  Number of Streams/Threads: 8
  Library: ./artifacts/simd/libturboquant.so

Initializing TurboQuant context...
  Type: SIMD Single

Saving context...
Updated .env: CONTEXT_PATH=./artifacts/turboquant_ctx_simd_128d_3b.bin

Cleaning up...
Context cleaned up

============================================================
Initialization completed successfully!
Context file: artifacts/turboquant_ctx_simd_128d_3b.bin
============================================================
```

### Context File Naming

Generated contexts follow the pattern:
```
turboquant_ctx_<variant>_<dimensions>d_<bit_width>b.bin
```

Examples:
- `turboquant_ctx_simd_128d_3b.bin` (CPU, 128 dims, 3 bits)
- `turboquant_ctx_simt-multi_128d_4b.bin` (CUDA batch, 128 dims, 4 bits)

---

## Running Inference

Once the environment is configured and context is initialized, you can run inference.

### Basic Usage (from main.py)

```python
from app.llama_models import LlamaCompressed, LlamaGenerator

# Initialize model with compressed KV cache
model = LlamaCompressed(
    max_seq_length=1024,    # Maximum sequence length
    batch_size=1,           # Batch size for inference
    device="cpu",           # Device: "cpu" or "cuda"
    is_batch=False,         # Enable batch processing mode
    bit_width=3,            # Override .env bit-width (optional)
    dims=128                # Override .env dimensions (optional)
)

# Create generator
generator = LlamaGenerator()

# Encode prompt
prompt = "What is the capital of France?"
prompt_tokens, prompt_tensors = model.input_encoding(prompt)

# Generate response
generated_tokens = generator.generate(prompt_tensors, model, max_gen_len=1024)

# Decode response
response = model.tokenizer.decode(generated_tokens)
print(response)
```

### Complete main.py Example

```python
from app.llama_models import LlamaCompressed, LlamaGenerator

if __name__ == '__main__':
    print("=" * 60)
    print("Llama Compressed Test")
    print("=" * 60)

    print("Initializing Llama compressed model...")
    # Uses configuration from .env
    model = LlamaCompressed(
        max_seq_length=1024,
        batch_size=1,
        device="cpu",       # "cpu" or "cuda"
        is_batch=False      # batch mode for parallel processing
    )

    generator = LlamaGenerator()
    prompt = "What is the capital of France ?"
    print(f"Prompt: {prompt}")

    print("Encoding tokens...")
    prompt_tokens, prompt_tensors = model.input_encoding(prompt)

    print("Generating answer...")
    generated_tokens = generator.generate(prompt_tensors, model)
    
    # Decode and print
    answer = model.tokenizer.decode(generated_tokens)
    print(f"Answer: {answer}")
```

### Device Selection Logic

When `TURBOQUANT_VARIANT=auto`:

| device | is_batch | Selected Variant |
|--------|----------|------------------|
| "cpu"  | False    | simd             |
| "cpu"  | True     | simd-multi       |
| "cuda" | False    | simt             |
| "cuda" | True     | simt-multi       |

Override with explicit variant:
```python
model = LlamaCompressed(
    max_seq_length=1024,
    batch_size=1,
    device="cpu",
    is_batch=True  # Will use simd-multi even on CPU
)
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LIB_PATH` | No | `""` | Direct path to libturboquant.so |
| `CONTEXT_PATH` | No | `""` | Path to initialized context file |
| `TURBOQUANT_ROOT` | No | `../TurboQuantQuantization` | TurboQuant source location |
| `TURBOQUANT_VARIANT` | No | `auto` | Variant selection |
| `DEFAULT_BIT_WIDTH` | No | `3` | Quantization bits (2-8) |
| `DEFAULT_DIMENSIONS` | No | `128` | Block dimension size |
| `DEFAULT_BLOCK_SIZE` | No | `8` | Threads/streams for batch |
| `PARAMS_PATH` | Yes | `./model/params.json` | Model hyperparameters |
| `CHECKPOINT_PATH` | Yes | `./model/consolidated.00.pth` | Model weights |
| `TOKENIZER_PATH` | Yes | `./model/tokenizer.model` | Tokenizer file |

### Variant Aliases

| Alias | Maps To | Use Case |
|-------|---------|----------|
| `cpu` | `simd` | Single-threaded CPU |
| `cpu-batch`, `cpu-multi` | `simd-multi` | Multi-threaded CPU |
| `cuda` | `simt` | Single-stream CUDA |
| `cuda-batch` | `simt-multi` | Multi-stream CUDA |

### API Differences by Variant

The Python wrappers handle these automatically, but for reference:

**SIMD (CPU) - Global State:**
```c
uint8_t turboquant_init(size_t dim, uint8_t bit_width);
void turboquant_clean(void);
uint8_t turboquant_save(const char* filename);
```

**SIMT (CUDA) - Context Pointer:**
```c
uint8_t turboquant_init(turboquant_context_t** ctx, size_t dim, uint8_t bit_width);
void turboquant_clean(turboquant_context_t* ctx);
void turboquant_context_destroy(turboquant_context_t** ctx);
uint8_t turboquant_save(turboquant_context_t* ctx, const char* filename);
```

---

## Troubleshooting

### "TurboQuant library not found"

**Cause**: Library not built or LIB_PATH incorrect

**Solution**:
```bash
# Rebuild
./INSTALL.sh

# Verify path in .env
ls artifacts/simd/libturboquant.so
```

### "turboquant_init failed with code X"

**Cause**: Context parameters don't match library build or corrupted context

**Solution**:
```bash
# Regenerate context
python -m scripts.initialize_context
```

### "undefined symbol: turboquant_context_destroy"

**Cause**: Mismatched API - trying to use SIMT API on SIMD library

**Solution**: Ensure LIB_PATH matches the variant you want:
```ini
# Wrong - SIMD library doesn't have context_destroy
LIB_PATH=./artifacts/simd/libturboquant.so

# Correct for your needs
LIB_PATH=./artifacts/simd/libturboquant.so  # Uses global state API
# or
LIB_PATH=./artifacts/simt/libturboquant.so    # Uses context pointer API
```

The `initialize_context.py` script automatically detects the correct API based on the library path.

### Model device mismatch

**Cause**: Model on CPU but trying to use CUDA variant

**Solution**: Match device parameter to variant:
```python
# CPU variant with CPU device
model = LlamaCompressed(..., device="cpu")  # Use with LIB_PATH=.../simd/...

# CUDA variant with CUDA device
model = LlamaCompressed(..., device="cuda") # Use with LIB_PATH=.../simt/...
```

### Context file version mismatch

**Cause**: Context generated with different bit-width/dimensions than model expects

**Solution**: Ensure `.env` settings match when generating context and running:
```ini
DEFAULT_BIT_WIDTH=3        # Must match between init and runtime
DEFAULT_DIMENSIONS=128     # Must match model head dimension
```

---

## Quick Start Checklist

1. [ ] Clone repository
2. [ ] Install dependencies: `pip install -r requirements.txt`
3. [ ] Build variants: `./INSTALL.sh`
4. [ ] Create `.env` file with model paths and variant selection
5. [ ] Place model files in `model/` directory
6. [ ] Run initialization: `python -m scripts.initialize_context`
7. [ ] Verify `CONTEXT_PATH` updated in `.env`
8. [ ] Run inference: `python -m app.main`

---

## Advanced Configuration

### Multiple Contexts for Different Configurations

You can maintain multiple contexts for different bit-widths:

```bash
# Generate 2-bit context
export DEFAULT_BIT_WIDTH=2
python -m scripts.initialize_context
mv artifacts/turboquant_ctx_simd_128d_2b.bin artifacts/contexts/

# Generate 4-bit context
export DEFAULT_BIT_WIDTH=4
python -m scripts.initialize_context
mv artifacts/turboquant_ctx_simd_128d_4b.bin artifacts/contexts/
```

Then select in `.env`:
```ini
CONTEXT_PATH=./artifacts/contexts/turboquant_ctx_simd_128d_4b.bin
```

### Programmatic Configuration

You can override .env settings in code:

```python
import os

# Override before importing app
os.environ["TURBOQUANT_VARIANT"] = "simd-multi"
os.environ["DEFAULT_BIT_WIDTH"] = "4"

from app.llama_models import LlamaCompressed

model = LlamaCompressed(...)
```

---

## Summary

1. **Build** all variants with `./INSTALL.sh`
2. **Configure** `.env` with model paths and variant selection
3. **Initialize** context with `python -m scripts.initialize_context`
4. **Run** inference with `python -m app.main`

The `.env` file is the central configuration point. The `LIB_PATH` takes highest priority for library selection, followed by `TURBOQUANT_VARIANT`. The initialization script auto-detects the API type (SIMD vs SIMT) based on the library path and uses the appropriate calling convention.
