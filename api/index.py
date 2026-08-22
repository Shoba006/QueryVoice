import os
import sys
from pathlib import Path

# Ensure read-only Lambda environment uses /tmp for caches
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TORCH_HOME"] = "/tmp/torch"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.server.main import app

# Export FastAPI app for Vercel Serverless Functions
app = app

