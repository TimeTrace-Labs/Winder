"""Test-collection setup: makes `scripts/pretrain.py` importable as a plain module (`import
pretrain`) so its `main(argv)` can be called in-process, without turning `scripts/` into a
package -- it stays a directory of standalone CLI drivers, exactly like `scripts/accept.py`/
`scripts/operator_smoke.py`, neither of which needed to be importable before this phase's tests
needed to call `pretrain.main(argv)` directly (in-process, not a subprocess) to check its
in-memory return value and the objects it builds along the way.
"""

import os
import sys

_SCRIPTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
