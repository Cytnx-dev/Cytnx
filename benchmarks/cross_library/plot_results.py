"""Plot per-step time and peak memory vs. chain length L and bond dimension
chi, grouped by (algorithm, library, backend), from the CSVs written by
run_all.py / the individual benchmark scripts.

Requires pandas and matplotlib (not part of this suite's core dependencies,
since plotting is a post-processing step, not part of the benchmark itself).

Usage:
    python plot_results.py                  # reads results/*.csv, writes results/plots/*.png
    python plot_results.py --results-dir results --out-dir results/plots
"""
import argparse
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt


def load_results(results_dir):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(results_dir, "*.csv")))]
    if not frames:
        raise FileNotFoundError(f"no CSV files found in {results_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["series"] = df["library"] + "/" + df["backend"]
    return df


def _plot_metric(df, algorithm, metric, x, fixed, fixed_value, ylabel, out_path):
    subset = df[(df["algorithm"] == algorithm) & (df[fixed] == fixed_value)]
    if subset.empty:
        return False
    fig, ax = plt.subplots()
    for series, group in subset.groupby("series"):
        group = group.sort_values(x)
        ax.plot(group[x], group[metric], marker="o", label=series)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{algorithm}: {ylabel} vs {x} ({fixed}={fixed_value})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def make_plots(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for algorithm in sorted(df["algorithm"].unique()):
        sub = df[df["algorithm"] == algorithm]
        # chi sweep at the smallest available L, and L sweep at the smallest available chi
        fixed_l = sub["L"].min()
        fixed_chi = sub["chi"].min()
        for metric, ylabel in [("step_time_sec", "time/step (s)"), ("peak_mem_mb", "peak memory (MB)")]:
            for x, fixed, fixed_value in [("chi", "L", fixed_l), ("L", "chi", fixed_chi)]:
                out_path = os.path.join(out_dir, f"{algorithm}_{metric}_vs_{x}.png")
                if _plot_metric(sub, algorithm, metric, x, fixed, fixed_value, ylabel, out_path):
                    written.append(out_path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "plots")

    df = load_results(args.results_dir)
    written = make_plots(df, out_dir)
    print(f"wrote {len(written)} plot(s) to {out_dir}/")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
