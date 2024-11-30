from __future__ import annotations

import os
import sys
from pathlib import Path


def prepare() -> Path:
    root_dir = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(root_dir))

    try:
        from build_full_mtc_example import as_needed
    except ImportError:
        print(
            "Please run this script from the "
            "activitysim/examples/example_estimation/notebooks directory."
        )
        raise

    as_needed(root_dir / "notebooks" / "test-estimation-data", 20_000)
    relative_path = os.path.relpath(
        root_dir
        / "notebooks"
        / "test-estimation-data"
        / "activitysim-prototype-mtc-extended"
    )
    os.chdir(relative_path)
    return relative_path
