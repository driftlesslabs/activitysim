from __future__ import annotations

import argparse
from pathlib import Path

from activitysim.examples.external import download_external_example


def main(download_dir):
    example_dir = download_external_example(
        name="prototype_mtc_extended",
        working_dir=download_dir,
        url="https://github.com/ActivitySim/activitysim-prototype-mtc/archive/refs/heads/extended.tar.gz",
        assets={
            "data_full.tar.zst": {
                "url": "https://github.com/ActivitySim/activitysim-prototype-mtc/releases/download/v1.3.4/data_full.tar.zst",
                "sha256": "b402506a61055e2d38621416dd9a5c7e3cf7517c0a9ae5869f6d760c03284ef3",
                "unpack": "data_full",
            },
            "test/prototype_mtc_reference_pipeline.zip": {
                "url": "https://github.com/ActivitySim/activitysim-prototype-mtc/releases/download/v1.3.2/prototype_mtc_extended_reference_pipeline.zip",
                "sha256": "4d94b6a8a83225dda17e9ca19c9110bc1df2df5b4b362effa153d1c8d31524f5",
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the prototype MTC extended example."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=Path.cwd(),
        help="Directory to download the example to. Defaults to the current working directory.",
    )
    args = parser.parse_args()
    main(Path(args.directory))
