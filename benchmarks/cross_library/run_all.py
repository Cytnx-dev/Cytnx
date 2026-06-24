"""Orchestration script: runs every library/algorithm benchmark in this
suite and collects all results under a single `results/` directory.

Each benchmark module is invoked in its own subprocess (rather than
imported in-process) so that one library's import side effects (global
device state, RNG seeding, monkeypatched backends) cannot leak into
another library's run, and so that a crash or hang in one benchmark does
not take down the rest of the sweep.

Usage:
    python run_all.py                      # run everything
    python run_all.py --only tenpy quimb    # restrict to a subset of libraries
    python run_all.py --skip cytnx          # skip the libraries that can't run here
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# (library, relative script path, output csv name, extra CLI args)
RUNS = [
    ("tenpy", "tenpy_bench/dmrg_dense.py", "tenpy_dmrg_dense.csv", []),
    ("tenpy", "tenpy_bench/dmrg_symmetric.py", "tenpy_dmrg_symmetric.csv", []),
    ("tenpy", "tenpy_bench/tdvp.py", "tenpy_tebd.csv", []),
    ("tenpy", "tenpy_bench/variational_manual_grad.py", "tenpy_variational.csv", []),
    ("quimb", "quimb_bench/dmrg_dense.py", "quimb_dmrg_dense.csv", []),
    ("quimb", "quimb_bench/dmrg_symmetric.py", "quimb_dmrg_symmetric.csv", []),
    ("quimb", "quimb_bench/tebd.py", "quimb_tebd.csv", []),
    ("quimb", "quimb_bench/variational_ad.py", "quimb_variational_ad.csv", ["--backend", "both"]),
    ("cytnx", "cytnx_bench/dmrg_dense.py", "cytnx_dmrg_dense.csv", []),
    ("cytnx", "cytnx_bench/dmrg_symmetric.py", "cytnx_dmrg_symmetric.csv", []),
    ("cytnx", "cytnx_bench/tebd.py", "cytnx_tebd.csv", []),
    ("cytnx", "cytnx_bench/variational_manual_grad.py", "cytnx_variational.csv", []),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=["tenpy", "quimb", "cytnx"],
                         help="run only these libraries (default: all)")
    parser.add_argument("--skip", nargs="*", choices=["tenpy", "quimb", "cytnx"], default=[],
                         help="skip these libraries")
    args = parser.parse_args()

    libraries = set(args.only) if args.only else {"tenpy", "quimb", "cytnx"}
    libraries -= set(args.skip)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    failures = []
    for library, script, out_name, extra_args in RUNS:
        if library not in libraries:
            continue
        script_path = os.path.join(HERE, script)
        out_csv = os.path.join(RESULTS_DIR, out_name)
        cmd = [sys.executable, script_path, out_csv] + extra_args
        print(f"=== running {script} -> {out_name} ===")
        result = subprocess.run(cmd, cwd=HERE)
        if result.returncode != 0:
            failures.append(script)
            print(f"!!! {script} failed with exit code {result.returncode}")

    if failures:
        print(f"\n{len(failures)} benchmark(s) failed: {failures}")
        sys.exit(1)
    print(f"\nAll requested benchmarks finished. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
