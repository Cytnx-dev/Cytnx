#!/usr/bin/env python3
"""Process Google Benchmark JSON from the L2 Float64 CUDA get experiment.

Only real-time medians are compared. Missing or skipped benchmarks remain
explicitly invalid; no fabricated latency is substituted.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

NAME_RE = re.compile(
    r"^GPUGetL2/([^/]+)/T(\d+)/U(\d+)/(Identity|Strided|Reversed|Random)/S([0-4])(?:/|$)"
)
PILOT_CASES = {
    "0": ("A262144", 262144, 1),
    "1": ("B", 2048, 32),
    "2": ("A262144", 256, 1024),
    "3": ("B", 4, 65536),
}


def records(path: Path):
    data = json.loads(path.read_text())
    groups = defaultdict(list)
    for entry in data.get("benchmarks", []):
        match = NAME_RE.match(entry.get("name", ""))
        if not match:
            continue
        family, units, nunit, pattern, strategy = match.groups()
        key = (family, int(units), int(nunit), pattern, int(strategy))
        if entry.get("error_occurred"):
            groups[key].append((None, entry.get("error_message", "benchmark error")))
            continue
        val = entry.get("real_time")
        if val is None or not isinstance(val, (float, int)) or val <= 0 or not math.isfinite(val):
            continue
        unit = entry.get("time_unit", "ns")
        convert = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
        if unit not in convert:
            raise ValueError(f"Unsupported Google Benchmark time unit: {unit}")
        # Prefer explicit reported median aggregate. Repetitions are kept for fallback.
        if entry.get("aggregate_name") not in (None, "median"):
            continue
        groups[key].append((float(val) * convert[unit],
                            entry.get("aggregate_name") or "repetition"))
    out = {}
    for key, pairs in groups.items():
        errors = [reason for value, reason in pairs if value is None]
        medians = [value for value, kind in pairs if value is not None and kind == "median"]
        samples = [value for value, kind in pairs if value is not None and kind == "repetition"]
        if errors:
            out[key] = (None, "; ".join(errors))
        elif medians:
            out[key] = (medians[0], "ok")
        elif len(samples) >= 5:
            out[key] = (statistics.median(samples), "ok")
        else:
            out[key] = (None, f"insufficient repetitions: {len(samples)}")
    return out


def choose_dispatch(path: Path, output: Path) -> str:
    data = records(path)
    result = []
    for bucket in "0123":
        family, units, nunit = PILOT_CASES[bucket]
        # Fallback is a transparent heuristic, not an invented measurement.
        fallback = {"0": 0, "1": 2, "2": 2, "3": 3}[bucket]
        options = {}
        for strategy in range(4):
            latency, status = data.get((family, units, nunit, "Identity", strategy),
                                       (None, "missing"))
            if status == "ok":
                options[strategy] = latency
        result.append(str(min(options, key=options.get) if len(options) == 4 else fallback))
    policy = "".join(result)
    output.write_text(policy + "\n")
    print(f"CYTNX_BENCH_GPU_GET_DISPATCH={policy} (fixed buckets <=4, <=32, <=1024, >1024)")
    return policy


def write_report(path: Path, output_dir: Path, policy: str = "") -> None:
    data = records(path)
    all_cases = sorted({case[:4] for case in data}, key=lambda c: (c[0], c[1], c[2], c[3]))
    rows = []
    for family, units, nunit, pattern in all_cases:
        baseline, baseline_status = data.get((family, units, nunit, pattern, 0),
                                              (None, "missing"))
        for s in range(5):
            latency, status = data.get((family, units, nunit, pattern, s),
                                       (None, "missing"))
            speedup = (baseline / latency if baseline and latency else None)
            rows.append({
                "family": family, "TotalElem": units, "Nunit": nunit,
                "pattern": pattern, "strategy": f"S{s}",
                "latency_us": f"{latency:.5f}" if latency is not None else "",
                "speedup_vs_S0": f"{speedup:.4f}" if speedup is not None else "",
                "status": status,
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No recognized GPUGetL2 benchmarks in JSON")
    with (output_dir / "gpu_get_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Cytnx PR #1230 — GPU Float64 Get L2 measurements",
        "",
        "Source: Google Benchmark wall-clock/real-time medians (5 repetitions per case).",
        "Speedup = S0 latency / strategy latency. Larger is faster.",
        "",
        f"Adaptive S4 bucket policy: \`{policy or 'not supplied'}\` (each digit selects S0..S3).",
        "Buckets: Nunit <=4, <=32, <=1024, and >1024.",
        "Policy is trained only on the four pilot cases; verify against ALL full cases.",
        "",
        "| Family | TotalElem | Nunit | Pattern | Strategy | Latency (µs) | Speedup | Status |",
        "|:--|--:|--:|:--|:--|--:|--:|:--|",
    ]
    for r in rows:
        md.append("| " + " | ".join([r["family"], str(r["TotalElem"]), str(r["Nunit"]),
                     r["pattern"], r["strategy"], r["latency_us"] or "—",
                     r["speedup_vs_S0"] or "—", r["status"]]) + " |")
    md.extend(["", "## Trends and caution",
        "",
        "The results include per-call CUDA metadata allocation and H2D copies, "
        "not just kernel time. S0 calls the unmodified PR #1230 Double L2 function. "
        "All strategies are correctness checked against independent CPU output.",
        "",
        "Kernel-only speedups are NOT isolated by this experiment; the metadata overhead "
        "cannot be quantified uniquely without a separate dedicated profiling experiment. "
        "Treat small differences as potentially dominated by metadata transfer, "
        "allocator synchronization, and GPU clock variance.",
        "",
        "A strategy should only be merged after confirming reproducibility on "
        "multiple GPU architectures and that any improvement survives measurement noise.",
    ])
    # A compact objective list of winners, losers and ties for reviewer context.
    winners, losers, ties = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in rows:
        if r["strategy"] == "S0" or r["status"] != "ok" or not r["speedup_vs_S0"]:
            continue
        sp = float(r["speedup_vs_S0"])
        if sp > 1.03: winners[r["strategy"]] += 1
        elif sp < 0.97: losers[r["strategy"]] += 1
        else: ties[r["strategy"]] += 1
    md.extend(["", "## Win/loss counts (±3% tolerance)", "",
               "| Strategy | Faster | Slower | Within ±3% |",
               "|:--|--:|--:|--:|"])
    for s in ("S1", "S2", "S3", "S4"):
        md.append(f"| {s} | {winners[s]} | {losers[s]} | {ties[s]} |")
    (output_dir / "gpu_get_report.md").write_text("\n".join(md) + "\n")
    print(f"Saved {len(rows)} records to {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("json", type=Path)
    pilot.add_argument("--output", type=Path, default=Path("dispatch.txt"))
    report = sub.add_parser("report")
    report.add_argument("json", type=Path)
    report.add_argument("--output-dir", type=Path, default=Path("results"))
    report.add_argument("--policy", default="")
    args = ap.parse_args()
    if args.cmd == "pilot":
        choose_dispatch(args.json, args.output)
    else:
        write_report(args.json, args.output_dir, args.policy)
