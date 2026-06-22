import os
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
_backend = str(_root / "backend")
_project = str(_root)

# backend must precede project root so backend/app/ wins over root app/.
# Remove-then-reinsert because the editable install .pth file may have
# already added backend/ at a later position.
for _p in [_project, _backend]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# Provide minimal env vars so get_settings() doesn't fail during tests.
os.environ.setdefault("GROQ_API_KEY", "test-key-for-unit-tests")
