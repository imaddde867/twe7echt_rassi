"""lifesim: a small agent-based life simulation. 1 tick = 1 day."""
import sys

if sys.version_info < (3, 10):  # the code uses `X | None` annotations at runtime
    raise ImportError(
        f"lifesim needs Python 3.10 or newer, found {sys.version.split()[0]}. "
        "On a cluster, load a newer Python module first (CI runs 3.11)."
    )
