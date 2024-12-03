from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path

# suppress RuntimeWarning from xarray
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="xarray",
)


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
    return Path(relative_path)


def backup(filename: str | os.PathLike):
    """Create or restore from a backup copy of a file."""
    backup_filename = f"{filename}.bak"
    if Path(backup_filename).exists():
        shutil.copy(backup_filename, filename)
    else:
        shutil.copy(filename, backup_filename)
