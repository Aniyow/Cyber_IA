import sys
from pathlib import Path

# Lets the tests import the project as `src.*` when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
