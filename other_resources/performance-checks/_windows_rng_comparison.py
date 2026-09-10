"""Compare pinned RNG revisions sequentially on one native Windows x64 runner."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil

REVISIONS = {
    "original": "72902ca9f951651adc6fcf7b9babe4a9ea3df29c",
    "owned": "ac76cb89e57d275f6e907865f9860ff7fec960ed",
    "fixed": "507c08aff",
}
SCRIPTS = ("fast-channel-random", "eet-random-scaling")


def write_json(path, data):
    """Flush progress after each process so failed jobs retain useful artifacts."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_process(command, checkout, env, destination):
    """Record process timing and Windows peak working set, including worker samples.

    Windows peak_wset is the OS-maintained lifetime high-water mark as last
    observed before exit. The process-tree RSS sum is sampled every 50 ms; short
    peaks between samples may be missed. Neither metric replaces the scripts'
    persistent-state and traced-allocation measurements.
    """
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak_wset = sampled_tree_rss = 0
    with (destination / "console.log").open("w", encoding="utf-8") as log:
        process = psutil.Popen(
            command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        while process.poll() is None:
            try:
                memory = process.memory_info()
                peak_wset = max(peak_wset, memory.peak_wset)
                total = memory.rss
                for child in process.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                sampled_tree_rss = max(sampled_tree_rss, total)
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.05)
        returncode = process.wait()
    record = {
        "command": command,
        "elapsed_seconds": time.perf_counter() - started,
        "returncode": returncode,
        "observed_peak_working_set_mib": peak_wset / 2**20,
        "sampled_peak_process_tree_rss_mib": sampled_tree_rss / 2**20,
    }
    write_json(destination / "resources.json", record)
    if returncode:
        print((destination / "console.log").read_text(encoding="utf-8")[-12000:])
        raise RuntimeError(f"Process failed ({returncode}): {command}")
    return record


def main():
    """Use the same interpreter, fixed sources, and both run orders for comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("These benchmarks require native Windows x64 Python")
    repo = Path(__file__).resolve().parents[2]
    metadata = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "logical_cpus": psutil.cpu_count(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "numba", "llvmlite", "pandas", "psutil", "cffi")
        },
        "revisions": {},
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        },
    }
    (output / "pip-freeze.txt").write_text(
        subprocess.check_output(
            ["uv", "pip", "freeze", "--python", sys.executable], text=True
        ),
        encoding="utf-8",
    )
    checkouts = {}
    for label, revision in REVISIONS.items():
        sha = subprocess.check_output(
            ["git", "rev-parse", revision], cwd=repo, text=True
        ).strip()
        checkout = output / "checkouts" / label
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(checkout), sha],
            cwd=repo,
            check=True,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(checkout)
        # Verify editable-install hooks cannot silently select another revision.
        source = subprocess.check_output(
            [sys.executable, "-c", "import activitysim; print(activitysim.__file__)"],
            cwd=checkout,
            env=env,
            text=True,
        ).strip()
        if not Path(source).resolve().is_relative_to(checkout):
            raise RuntimeError(f"Wrong source imported for {label}: {source}")
        metadata["revisions"][label] = {"sha": sha, "imported_source": source}
        checkouts[label] = checkout, env
    write_json(output / "metadata.json", metadata)
    runs = []
    # Run correctness first, outside benchmark processes and timing windows.
    for label, (checkout, env) in checkouts.items():
        tests = [
            str(path.relative_to(checkout))
            for name in (
                "test_fast_random.py",
                "test_fast_channel.py",
                "test_fast_entropy.py",
                "test_random.py",
                "test_random_imports.py",
            )
            if (path := checkout / "activitysim" / "core" / "test" / name).exists()
        ]
        print(f"TEST {label}", flush=True)
        run_process(
            [sys.executable, "-m", "pytest", *tests, "-q", "-o", "addopts="],
            checkout,
            env,
            output / "tests" / label,
        )
    for round_number, order in enumerate(
        [list(REVISIONS), list(reversed(REVISIONS))], start=1
    ):
        for script in SCRIPTS:
            for label in order:
                checkout, env = checkouts[label]
                destination = output / f"round-{round_number}" / label / script
                command = [
                    sys.executable,
                    f"other_resources/performance-checks/{script}.py",
                    "--profile",
                    "full",
                    "--repeat",
                    "15" if script == "eet-random-scaling" else "5",
                    "--output-dir",
                    str(destination),
                ]
                if script == "eet-random-scaling":
                    command.append("--skip-plots")
                print(f"START round {round_number}: {label} {script}", flush=True)
                record = run_process(command, checkout, env, destination)
                result = json.loads((destination / "results.json").read_text())
                if not all(c["passed"] for c in result["invariance_checks"]):
                    raise RuntimeError(f"Stream invariant failed: {destination}")
                record.update(round=round_number, revision=label, script=script)
                runs.append(record)
                write_json(output / "runs.json", runs)
                print(f"DONE {record}", flush=True)
    summary = [
        "# Native Windows x64 RNG comparison",
        "",
        "All benchmark processes completed and all stream invariants passed.",
        "See artifacts for individual samples, tests and environment metadata.",
        "",
        "| Round | Revision | Script | Seconds | Observed peak working set MiB |",
        "|---|---|---|---:|---:|",
    ]
    for run in runs:
        summary.append(
            f"| {run['round']} | {run['revision']} | {run['script']} | "
            f"{run['elapsed_seconds']:.2f} | "
            f"{run['observed_peak_working_set_mib']:.2f} |"
        )
    report = "\n".join(summary) + "\n"
    (output / "summary.md").write_text(report, encoding="utf-8")
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(report)


if __name__ == "__main__":
    main()
