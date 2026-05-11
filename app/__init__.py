import sys
from pathlib import Path

# Get absolute path to project root (parent of app/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Add llama-models to sys.path for absolute imports
sys.path.insert(0, str(PROJECT_ROOT / 'model' / 'llama-models'))

# Path constants as Path objects for consistency
PARAMS_PATH = PROJECT_ROOT / "model" / "params.json"
CHECKPOINT_PATH = PROJECT_ROOT / "model" / "consolidated.00.pth"
TOKENIZER_PATH = PROJECT_ROOT / "model" / "tokenizer.model"
