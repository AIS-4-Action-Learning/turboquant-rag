import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Get absolute path to project root (parent of app/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Load environment variables from .env file
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

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

# TurboQuant parameters
DEFAULT_BIT_WIDTH = int(os.getenv("DEFAULT_BIT_WIDTH", _TURBOQUANT_BIT_WIDTH))
DEFAULT_BLOCK_SIZE = int(os.getenv("DEFAULT_BLOCK_SIZE", _TURBOQUANT_BLOCK_SIZE))
DEFAULT_DIMENSIONS = int(os.getenv("DEFAULT_DIMENSIONS", _TURBOQUANT_DIMENSIONS))
