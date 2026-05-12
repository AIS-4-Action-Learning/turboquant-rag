#!/usr/bin/env python3
"""
Script to initialize TurboQuant context based on parameters from .env file.
This script calls the TurboQuant C API directly via ctypes:
1. Reads parameters from .env file
2. Loads the TurboQuant shared library
3. Calls turboquant_init_batch to create context
4. Calls turboquant_batch_save to save context to artifacts/
5. Cleans up with turboquant_batch_destroy
"""

import ctypes
import os
import re
import sys
from pathlib import Path

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv


# ctypes structures mirroring turboquant.h
class TurboQuantContext(ctypes.Structure):
    """Mirrors turboquant_context_t from turboquant.h"""
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


class TurboQuantBatchContext(ctypes.Structure):
    """Mirrors turboquant_batch_context_t from turboquant.h"""
    _fields_ = [
        ("contexts", ctypes.POINTER(ctypes.POINTER(TurboQuantContext))),
        ("n_streams", ctypes.c_uint8),
        ("dims", ctypes.c_size_t),
        ("bit_width", ctypes.c_uint8),
        ("is_init", ctypes.c_uint8),
    ]


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def load_env_vars():
    """Load environment variables from .env file."""
    project_root = get_project_root()
    env_path = project_root / ".env"
    load_dotenv(env_path, override=True)
    
    # Default dimension is 128, default bit_width is 3
    # DEFAULT_BLOCK_SIZE in .env is actually the number of streams for batch
    return {
        "bit_width": int(os.getenv("DEFAULT_BIT_WIDTH", "3")),
        "dims": int(os.getenv("DEFAULT_DIMENSIONS", "128")),
        "n_streams": int(os.getenv("DEFAULT_BLOCK_SIZE", "8")),  # batch/stream count
        "lib_path": os.getenv("LIB_PATH", ""),
        "turboquant_root": Path(os.getenv("TURBOQUANT_ROOT", "../TurboQuantQuantization")),
    }


def find_library(lib_path: str) -> Path:
    """Find the TurboQuant shared library."""
    if lib_path:
        p = Path(lib_path)
        if p.exists():
            return p
    
    # Search only in artifacts folder
    artifacts_dir = get_project_root() / "artifacts"
    search_paths = [
        artifacts_dir / "libturboquant.so",
        artifacts_dir / "libturboquant.dylib",
        artifacts_dir / "turboquant.dll",
    ]
    
    for p in search_paths:
        if p.exists():
            return p
    
    raise FileNotFoundError(
        f"TurboQuant library not found. Searched in:\n"
        f"  - LIB_PATH env var: {lib_path}\n"
        f"  - {artifacts_dir}"
    )


def initialize_context(lib_path: Path, dims: int, bit_width: int, n_streams: int) -> ctypes.POINTER:
    """Initialize TurboQuant batch context using ctypes."""
    lib = ctypes.CDLL(str(lib_path))
    
    # Setup function signatures
    lib.turboquant_batch_init.argtypes = [
        ctypes.POINTER(ctypes.POINTER(TurboQuantBatchContext)),
        ctypes.c_size_t,
        ctypes.c_uint8,
        ctypes.c_uint8,
    ]
    lib.turboquant_batch_init.restype = ctypes.c_uint8
    
    lib.turboquant_batch_save.argtypes = [
        ctypes.POINTER(TurboQuantBatchContext),
        ctypes.c_char_p,
    ]
    lib.turboquant_batch_save.restype = ctypes.c_uint8
    
    lib.turboquant_batch_destroy.argtypes = [
        ctypes.POINTER(ctypes.POINTER(TurboQuantBatchContext))
    ]
    lib.turboquant_batch_destroy.restype = None
    
    # Allocate batch context pointer
    batch_ctx = ctypes.POINTER(TurboQuantBatchContext)()
    
    # Initialize
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
    
    return lib, batch_ctx


def save_context(lib, batch_ctx: ctypes.POINTER, output_path: Path) -> None:
    """Save the context to a file."""
    status = lib.turboquant_batch_save(
        batch_ctx,
        str(output_path).encode("utf-8")
    )
    
    if status != 0:
        raise RuntimeError(f"turboquant_batch_save failed with code {status}")
    
    print(f"Context saved to: {output_path}")


def format_env_context_path(context_path: Path, project_root: Path) -> str:
    """Format context path for .env usage (prefer project-relative path)."""
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


def destroy_context(lib, batch_ctx: ctypes.POINTER) -> None:
    """Destroy the batch context."""
    lib.turboquant_batch_destroy(ctypes.byref(batch_ctx))
    print("Context destroyed")


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
    turboquant_root = env_vars["turboquant_root"]
    
    print(f"\nConfiguration:")
    print(f"  TurboQuant Root: {turboquant_root}")
    print(f"  Bit Width: {bit_width}")
    print(f"  Dimensions: {dims}")
    print(f"  Number of Streams: {n_streams}")
    
    # Find library
    try:
        lib_path = find_library(lib_path_str)
        print(f"  Library: {lib_path}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease place libturboquant.so in artifacts/ or set LIB_PATH in .env")
        sys.exit(1)
    
    # Setup output path
    project_root = get_project_root()
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    output_path = artifacts_dir / f"turboquant_ctx_{dims}d_{bit_width}b.bin"
    
    try:
        # Initialize context
        print(f"\nInitializing TurboQuant batch context...")
        lib, batch_ctx = initialize_context(lib_path, dims, bit_width, n_streams)
        print("Initialization successful")
        
        # Save context
        print(f"\nSaving context...")
        save_context(lib, batch_ctx, output_path)
        update_env_context_path(output_path)
        
        # Cleanup
        print(f"\nCleaning up...")
        destroy_context(lib, batch_ctx)
        
        print("\n" + "=" * 60)
        print("Initialization completed successfully!")
        print(f"Context file: {output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
