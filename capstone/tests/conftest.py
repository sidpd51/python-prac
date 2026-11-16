"""Capstone test setup.

`study verify capstone` sets STUDY_SOLUTIONS=1: then `import docmind` resolves to the
reference implementation in solutions/capstone/src instead of YOUR code in capstone/src.
"""

import os
import sys
from pathlib import Path

if os.environ.get("STUDY_SOLUTIONS") == "1":
    ref = Path(__file__).resolve().parents[2] / "solutions" / "capstone" / "src"
    sys.path.insert(0, str(ref))
    for name in [m for m in sys.modules if m == "docmind" or m.startswith("docmind.")]:
        del sys.modules[name]
