#!/usr/bin/env python3
"""
Script to initialize TurboQuant context based on parameters from .env file.
Handles API differences between SIMD (global state) and SIMT (explicit context).
"""

import ctypes
import os
import re
import sys
from pathlib import Path

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def load_env_vars():
    """Load environment variables from .env file."""
    project_root = get_project_root()
    env_path = project_root / ".env"
    load_dotenv(env_path, override=True)
    
    return {
        "bit_width": int(os.getenv("DEFAULT_BIT_WIDTH", "3")) - 1,
        "dims": int(os.getenv("DEFAULT_DIMENSIONS", "128")),
        "n_streams": int(os.getenv("DEFAULT_BLOCK_SIZE", "8")),
        "variant": os.getenv("TURBOQUANT_VARIANT", "auto"),
        "lib_path": os.getenv("LIB_PATH", ""),
    }


def normalize_variant(variant: str) -> str:
    """Normalize variant name to canonical form."""
    variant = variant.lower()
    variant_map = {
        "simt-batch": "simt-multi",
        "cuda-batch": "simt-multi",
        "cuda": "simt",
        "cpu": "simd",
        "cpu-batch": "simd-multi",
        "cpu-multi": "simd-multi",
    }
    return variant_map.get(variant, variant)


def is_simd_variant(variant: str) -> bool:
    """Check if variant uses SIMD API (global state, no context)."""
    return variant.startswith("simd")


def is_batch_variant(variant: str) -> bool:
    """Check if variant uses batch API."""
    return "-multi" in variant


def find_library(lib_path_str: str, variant: str) -> tuple[Path, str]:
    """Find the TurboQuant shared library using variant-aware search.
    Returns (lib_path, detected_variant).
    """
    # Priority 1: LIB_PATH from .env takes precedence
    if lib_path_str:
        p = Path(lib_path_str)
        if p.exists():
            # Extract variant from path if possible
            # Check longer variant names first to avoid partial matches (e.g., simt matching simt-multi)
            for v in ["simd-multi", "simt-multi", "simd", "simt"]:
                if v in str(p):
                    return p, v
            return p, normalize_variant(variant)

    # Priority 2: Use variant-based search
    if variant == "auto":
        variant_name = "simd"  # Default to SIMD for CPU
    else:
        variant_name = normalize_variant(variant)

    artifacts_dir = get_project_root() / "artifacts"
    lib_path = artifacts_dir / variant_name / "libturboquant.so"

    if lib_path.exists():
        return lib_path, variant_name

    # Fallback: search all variant directories
    available = []
    fallback_path = None
    for variant_dir in artifacts_dir.iterdir():
        if variant_dir.is_dir():
            lib_file = variant_dir / "libturboquant.so"
            if lib_file.exists():
                available.append(variant_dir.name)
                if fallback_path is None:
                    fallback_path = lib_file

    if fallback_path is not None:
        print(f"Warning: Requested variant '{variant_name}' not found, using '{fallback_path.parent.name}'")
        return fallback_path, fallback_path.parent.name

    raise FileNotFoundError(
        f"TurboQuant library not found. Searched in:\n"
        f"  - LIB_PATH env var: {lib_path_str}\n"
        f"  - artifacts/{variant_name}/libturboquant.so\n"
        f"  Available variants: {available}\n"
        f"Run INSTALL.sh to build all variants."
    )


# ============================================================================
# SIMD Single Context API (global state)
# ============================================================================

def setup_simd_single_api(lib: ctypes.CDLL) -> None:
    """Setup function signatures for SIMD single context."""
    lib.turboquant_init.argtypes = [
        ctypes.c_size_t,  # dim
        ctypes.c_uint8,   # bit_width
    ]
    lib.turboquant_init.restype = ctypes.c_uint8
    
    # turboquant_clean takes no arguments
    lib.turboquant_clean.argtypes = []
    lib.turboquant_clean.restype = None
    
    # save/load take only filename
    lib.turboquant_save.argtypes = [ctypes.c_char_p]
    lib.turboquant_save.restype = ctypes.c_uint8
    
    lib.turboquant_init_load.argtypes = [ctypes.c_char_p]
    lib.turboquant_init_load.restype = ctypes.c_uint8


def initialize_simd_single(lib: ctypes.CDLL, dims: int, bit_width: int) -> None:
    """Initialize SIMD single context (global state, no context pointer)."""
    setup_simd_single_api(lib)
    
    status = lib.turboquant_init(
        ctypes.c_size_t(dims),
        ctypes.c_uint8(bit_width),
    )
    
    if status != 0:
        raise RuntimeError(f"turboquant_init failed with code {status}")
    
    return None


def save_simd_single(lib: ctypes.CDLL, output_path: Path) -> None:
    """Save SIMD single context."""
    status = lib.turboquant_save(str(output_path).encode("utf-8"))
    if status != 0:
        raise RuntimeError(f"turboquant_save failed with code {status}")


def cleanup_simd_single(lib: ctypes.CDLL) -> None:
    """Cleanup SIMD single context (uses turboquant_clean, no destroy)."""
    lib.turboquant_clean()
    print("Context cleaned up")


# ============================================================================
# SIMT Single Context API (explicit context)
# ============================================================================

class TurboQuantContextSIMT(ctypes.Structure):
    """Mirrors turboquant_context_t from simt/turboquant.h"""
    _fields_ = [
        ("mse_quantizer", ctypes.c_void_p),
        ("mse_buffer", ctypes.c_void_p),
        ("y", ctypes.c_void_p),
        ("h_bstring", ctypes.POINTER(ctypes.c_uint8)),
        ("d_bstring", ctypes.POINTER(ctypes.c_uint8)),
        ("bstring_size", ctypes.c_size_t),
        ("h_qjl", ctypes.POINTER(ctypes.c_uint8)),
        ("d_qjl", ctypes.POINTER(ctypes.c_uint8)),
        ("qjl_size", ctypes.c_size_t),
        ("compute_stream", ctypes.c_void_p),
        ("is_init", ctypes.c_uint8),
    ]


def setup_simt_single_api(lib: ctypes.CDLL) -> None:
    """Setup function signatures for SIMT single context."""
    lib.turboquant_init.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # context pointer
        ctypes.c_size_t,
        ctypes.c_uint8,
    ]
    lib.turboquant_init.restype = ctypes.c_uint8
    
    lib.turboquant_clean.argtypes = [ctypes.c_void_p]
    lib.turboquant_clean.restype = None
    
    lib.turboquant_context_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.turboquant_context_destroy.restype = None
    
    lib.turboquant_save.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.turboquant_save.restype = ctypes.c_uint8


def initialize_simt_single(lib: ctypes.CDLL, dims: int, bit_width: int) -> ctypes.c_void_p:
    """Initialize SIMT single context (returns context pointer)."""
    setup_simt_single_api(lib)
    
    ctx = ctypes.c_void_p()
    
    status = lib.turboquant_init(
        ctypes.byref(ctx),
        ctypes.c_size_t(dims),
        ctypes.c_uint8(bit_width),
    )
    
    if status != 0:
        raise RuntimeError(f"turboquant_init failed with code {status}")
    
    return ctx


def save_simt_single(lib: ctypes.CDLL, ctx: ctypes.c_void_p, output_path: Path) -> None:
    """Save SIMT single context."""
    status = lib.turboquant_save(ctx, str(output_path).encode("utf-8"))
    if status != 0:
        raise RuntimeError(f"turboquant_save failed with code {status}")


def cleanup_simt_single(lib: ctypes.CDLL, ctx: ctypes.c_void_p) -> None:
    """Cleanup SIMT single context (clean then destroy)."""
    lib.turboquant_clean(ctx)
    lib.turboquant_context_destroy(ctypes.byref(ctx))
    print("Context destroyed")


# ============================================================================
# SIMD Batch Context API
# ============================================================================

class TurboQuantBatchCtxSIMD(ctypes.Structure):
    """Mirrors turboquant_batch_ctx_t from simd-multi/turboquant.h"""
    _fields_ = [
        ("quantizer", ctypes.c_void_p),
        ("threads", ctypes.c_void_p),
        ("n_threads", ctypes.c_size_t),
        ("dims", ctypes.c_size_t),
        ("bit_width", ctypes.c_uint8),
        ("is_init", ctypes.c_uint8),
    ]


def setup_simd_batch_api(lib: ctypes.CDLL) -> None:
    """Setup function signatures for SIMD batch context."""
    lib.turboquant_batch_init.argtypes = [
        ctypes.POINTER(ctypes.POINTER(TurboQuantBatchCtxSIMD)),
        ctypes.c_size_t,  # dims
        ctypes.c_uint8,   # bit_width
        ctypes.c_size_t,  # n_threads
    ]
    lib.turboquant_batch_init.restype = ctypes.c_uint8
    
    lib.turboquant_batch_destroy.argtypes = [
        ctypes.POINTER(ctypes.POINTER(TurboQuantBatchCtxSIMD))
    ]
    lib.turboquant_batch_destroy.restype = None
    
    lib.turboquant_batch_save.argtypes = [
        ctypes.POINTER(TurboQuantBatchCtxSIMD),
        ctypes.c_char_p,
    ]
    lib.turboquant_batch_save.restype = ctypes.c_uint8


def initialize_simd_batch(lib: ctypes.CDLL, dims: int, bit_width: int, n_threads: int):
    """Initialize SIMD batch context."""
    setup_simd_batch_api(lib)
    
    batch_ctx = ctypes.POINTER(TurboQuantBatchCtxSIMD)()
    
    status = lib.turboquant_batch_init(
        ctypes.byref(batch_ctx),
        ctypes.c_size_t(dims),
        ctypes.c_uint8(bit_width),
        ctypes.c_size_t(n_threads),
    )
    
    if status != 0:
        raise RuntimeError(f"turboquant_batch_init failed with code {status}")
    
    if not batch_ctx or not batch_ctx.contents.is_init:
        raise RuntimeError("Failed to initialize TurboQuant batch context")
    
    return batch_ctx


def save_simd_batch(lib, batch_ctx, output_path: Path) -> None:
    """Save SIMD batch context."""
    status = lib.turboquant_batch_save(batch_ctx, str(output_path).encode("utf-8"))
    if status != 0:
        raise RuntimeError(f"turboquant_batch_save failed with code {status}")


def cleanup_simd_batch(lib, batch_ctx) -> None:
    """Cleanup SIMD batch context."""
    lib.turboquant_batch_destroy(ctypes.byref(batch_ctx))
    print("Batch context destroyed")


# ============================================================================
# SIMT Batch Context API
# ============================================================================

class TurboQuantContextSIMTStruct(ctypes.Structure):
    """Mirrors turboquant_context_t from simt/turboquant.h"""
    _fields_ = [
        ("mse_quantizer", ctypes.c_void_p),
        ("mse_buffer", ctypes.c_void_p),
        ("y", ctypes.c_void_p),
        ("h_bstring", ctypes.POINTER(ctypes.c_uint8)),
        ("d_bstring", ctypes.POINTER(ctypes.c_uint8)),
        ("bstring_size", ctypes.c_size_t),
        ("h_qjl", ctypes.POINTER(ctypes.c_uint8)),
        ("d_qjl", ctypes.POINTER(ctypes.c_uint8)),
        ("qjl_size", ctypes.c_size_t),
        ("compute_stream", ctypes.c_void_p),
        ("is_init", ctypes.c_uint8),
    ]


class TurboQuantBatchContextSIMT(ctypes.Structure):
    """Mirrors turboquant_batch_context_t from simt-multi/turboquant.h"""
    _fields_ = [
        ("contexts", ctypes.POINTER(ctypes.POINTER(TurboQuantContextSIMTStruct))),
        ("n_streams", ctypes.c_uint8),
        ("dims", ctypes.c_size_t),
        ("bit_width", ctypes.c_uint8),
        ("is_init", ctypes.c_uint8),
    ]


def setup_simt_batch_api(lib: ctypes.CDLL) -> None:
    """Setup function signatures for SIMT batch context."""
    lib.turboquant_batch_init.argtypes = [
        ctypes.POINTER(ctypes.POINTER(TurboQuantBatchContextSIMT)),
        ctypes.c_size_t,  # dim
        ctypes.c_uint8,   # bit_width
        ctypes.c_uint8,   # n_streams
    ]
    lib.turboquant_batch_init.restype = ctypes.c_uint8
    
    lib.turboquant_batch_destroy.argtypes = [
        ctypes.POINTER(ctypes.POINTER(TurboQuantBatchContextSIMT))
    ]
    lib.turboquant_batch_destroy.restype = None
    
    lib.turboquant_batch_save.argtypes = [
        ctypes.POINTER(TurboQuantBatchContextSIMT),
        ctypes.c_char_p,
    ]
    lib.turboquant_batch_save.restype = ctypes.c_uint8


def initialize_simt_batch(lib: ctypes.CDLL, dims: int, bit_width: int, n_streams: int):
    """Initialize SIMT batch context."""
    setup_simt_batch_api(lib)
    
    batch_ctx = ctypes.POINTER(TurboQuantBatchContextSIMT)()
    
    status = lib.turboquant_batch_init(
        ctypes.byref(batch_ctx),
        ctypes.c_size_t(dims),
        ctypes.c_uint8(bit_width),
        ctypes.c_uint8(n_streams),
    )
    
    if status != 0:
        raise RuntimeError(f"turboquant_batch_init failed with code {status}")
    
    if not batch_ctx or not batch_ctx.contents.is_init:
        raise RuntimeError("Failed to initialize TurboQuant batch context")
    
    return batch_ctx


def save_simt_batch(lib, batch_ctx, output_path: Path) -> None:
    """Save SIMT batch context."""
    status = lib.turboquant_batch_save(batch_ctx, str(output_path).encode("utf-8"))
    if status != 0:
        raise RuntimeError(f"turboquant_batch_save failed with code {status}")


def cleanup_simt_batch(lib, batch_ctx) -> None:
    """Cleanup SIMT batch context."""
    lib.turboquant_batch_destroy(ctypes.byref(batch_ctx))
    print("Batch context destroyed")


# ============================================================================
# High-level wrapper
# ============================================================================

def initialize_context(lib_path: Path, dims: int, bit_width: int, n_streams: int, variant: str):
    """Initialize TurboQuant context based on variant."""
    lib = ctypes.CDLL(str(lib_path))
    is_batch = is_batch_variant(variant)
    is_simd = is_simd_variant(variant)
    
    if is_batch:
        if is_simd:
            ctx = initialize_simd_batch(lib, dims, bit_width, n_streams)
        else:
            ctx = initialize_simt_batch(lib, dims, bit_width, n_streams)
    else:
        if is_simd:
            ctx = initialize_simd_single(lib, dims, bit_width)
        else:
            ctx = initialize_simt_single(lib, dims, bit_width)
    
    return lib, ctx, is_batch, is_simd


def save_context(lib, ctx, output_path: Path, is_batch: bool, is_simd: bool) -> None:
    """Save context based on variant."""
    if is_batch:
        if is_simd:
            save_simd_batch(lib, ctx, output_path)
        else:
            save_simt_batch(lib, ctx, output_path)
    else:
        if is_simd:
            save_simd_single(lib, output_path)
        else:
            save_simt_single(lib, ctx, output_path)


def cleanup_context(lib, ctx, is_batch: bool, is_simd: bool) -> None:
    """Cleanup context based on variant."""
    if is_batch:
        if is_simd:
            cleanup_simd_batch(lib, ctx)
        else:
            cleanup_simt_batch(lib, ctx)
    else:
        if is_simd:
            cleanup_simd_single(lib)
        else:
            cleanup_simt_single(lib, ctx)


# ============================================================================
# .env update helpers
# ============================================================================

def format_env_context_path(context_path: Path, project_root: Path) -> str:
    """Format context path for .env usage."""
    context_abs = context_path.resolve()
    try:
        rel = context_abs.relative_to(project_root.resolve())
        return f"./{rel.as_posix()}"
    except ValueError:
        return str(context_abs)


def update_env_context_path(context_path: Path) -> None:
    """Update CONTEXT_PATH in .env to point to the newly created context file."""
    project_root = get_project_root()
    env_path = project_root / ".env"
    env_context_path = format_env_context_path(context_path, project_root)

    lines = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    key = "CONTEXT_PATH"
    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    updated = False

    for idx, line in enumerate(lines):
        if key_pattern.match(line):
            lines[idx] = f"{key}={env_context_path}"
            updated = True
            break

    if not updated:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{key}={env_context_path}")

    env_path.write_text("\n".join(lines) + "\n")
    os.environ[key] = env_context_path
    print(f"Updated .env: {key}={env_context_path}")


def main():
    print("=" * 60)
    print("TurboQuant Context Initialization")
    print("=" * 60)
    
    # Load environment variables
    env_vars = load_env_vars()
    bit_width = env_vars["bit_width"]
    dims = env_vars["dims"]
    n_streams = env_vars["n_streams"]
    lib_path_str = env_vars["lib_path"]
    variant = env_vars["variant"]
    
    # Find library and detect variant from path
    try:
        lib_path, detected_variant = find_library(lib_path_str, variant)
        # Use detected variant if LIB_PATH specified a specific variant
        if variant == "auto" or not lib_path_str:
            variant = detected_variant
        else:
            variant = normalize_variant(variant)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Variant: {variant}")
    print(f"  Bit Width: {bit_width}")
    print(f"  Dimensions: {dims}")
    print(f"  Number of Streams/Threads: {n_streams}")
    print(f"  Library: {lib_path}")
    
    # Setup output path
    project_root = get_project_root()
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    output_path = artifacts_dir / f"turboquant_ctx_{variant}_{dims}d_{bit_width + 1}b.bin"
    
    try:
        # Initialize context
        print(f"\nInitializing TurboQuant context...")
        lib, ctx, is_batch, is_simd = initialize_context(lib_path, dims, bit_width, n_streams, variant)
        
        print(f"  Type: {'SIMD' if is_simd else 'SIMT'} {'Batch' if is_batch else 'Single'}")
        
        # Save context
        print(f"\nSaving context...")
        save_context(lib, ctx, output_path, is_batch, is_simd)
        update_env_context_path(output_path)
        
        # Cleanup
        print(f"\nCleaning up...")
        cleanup_context(lib, ctx, is_batch, is_simd)
        
        print("\n" + "=" * 60)
        print("Initialization completed successfully!")
        print(f"Context file: {output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
