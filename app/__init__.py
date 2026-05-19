import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Get absolute path to project root (parent of app/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Load environment variables from .env file
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

# Add llama-models to sys.path for absolute imports
sys.path.insert(0, str(PROJECT_ROOT / 'model' / 'llama-models'))

# TurboQuant defaults - used only when env vars not set
_TURBOQUANT_BIT_WIDTH = 3
_TURBOQUANT_BLOCK_SIZE = 8
_TURBOQUANT_DIMENSIONS = 128

# Path constants from environment variables
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", PROJECT_ROOT / "model" / "params.json"))
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", PROJECT_ROOT / "model" / "consolidated.00.pth"))
TOKENIZER_PATH = Path(os.getenv("TOKENIZER_PATH", PROJECT_ROOT / "model" / "tokenizer.model"))

CONTEXT_PATH = os.getenv("CONTEXT_PATH", "")
LIB_PATH = os.getenv("LIB_PATH", "")
TURBOQUANT_ROOT = Path(os.getenv("TURBOQUANT_ROOT", PROJECT_ROOT.parent / "TurboQuantQuantization"))

# TurboQuant variant configuration
TURBOQUANT_VARIANT = os.getenv("TURBOQUANT_VARIANT", "auto")

def get_turboquant_lib_path(device: str = "cuda", is_batch: bool = True) -> Path:
    """
    Get the appropriate TurboQuant library path based on variant configuration.
    
    Priority:
    1. If LIB_PATH is set in .env, use it directly
    2. If TURBOQUANT_VARIANT is explicitly set (not 'auto'), use that variant
    3. If TURBOQUANT_VARIANT='auto', select based on device and is_batch
    
    Variants:
    - simd: CPU single-threaded
    - simd-multi: CPU multi-threaded/batch
    - simt: CUDA single-stream
    - simt-multi (simt-batch): CUDA multi-stream/batch
    """
    # Priority 1: LIB_PATH from .env takes precedence
    if LIB_PATH:
        return Path(LIB_PATH)
    
    # Priority 2: Explicit variant selection
    variant = TURBOQUANT_VARIANT.lower()
    
    if variant == "auto":
        # Priority 3: Auto-detect based on device and batch mode
        if device == "cuda":
            variant_name = "simt-multi" if is_batch else "simt"
        else:  # cpu
            variant_name = "simd-multi" if is_batch else "simd"
    else:
        variant_name = variant
    
    # Map legacy naming
    variant_map = {
        "simt-batch": "simt-multi",
        "cuda-batch": "simt-multi",
        "cuda": "simt",
        "cpu": "simd",
        "cpu-batch": "simd-multi",
        "cpu-multi": "simd-multi",
    }
    variant_name = variant_map.get(variant_name, variant_name)
    
    lib_path = PROJECT_ROOT / "artifacts" / variant_name / "libturboquant.so"
    
    if not lib_path.exists():
        available = [p.name for p in (PROJECT_ROOT / "artifacts").iterdir() 
                     if (p / "libturboquant.so").exists()]
        raise FileNotFoundError(
            f"TurboQuant library not found at {lib_path}. "
            f"Available variants: {available}. "
            f"Run INSTALL.sh to build all variants."
        )
    
    return lib_path

# TurboQuant parameters
DEFAULT_BIT_WIDTH = float(os.getenv("DEFAULT_BIT_WIDTH", _TURBOQUANT_BIT_WIDTH))
DEFAULT_BLOCK_SIZE = int(os.getenv("DEFAULT_BLOCK_SIZE", _TURBOQUANT_BLOCK_SIZE))
DEFAULT_DIMENSIONS = int(os.getenv("DEFAULT_DIMENSIONS", _TURBOQUANT_DIMENSIONS))

# ============================================================================
# Compressor Factory - Import from appropriate module based on variant
# ============================================================================

def _get_compressor_factory():
    """Import and return the appropriate compressor factory function."""
    from app.turboquant_simd import get_compressor_for_variant
    return get_compressor_for_variant

# Make factory available at package level
get_compressor_for_variant = _get_compressor_factory()

# Backward compatibility
from app.turboquant_simd import TurboQuantCompressorBase
TurboQuantCompressor = TurboQuantCompressorBase
