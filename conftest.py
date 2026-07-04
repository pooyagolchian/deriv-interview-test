"""Ensure the repo root is importable so tests can `import evalharness` / `import validate`
without requiring an editable install."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
