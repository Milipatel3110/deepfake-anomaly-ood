#!/usr/bin/env python3
"""
compare_methods_plot.py
Make presentation-ready plots comparing ODIN / ΔConf / Energy / ΔEnergy / Mahalanobis / Combined.

Inputs:
  --main_csv  : path to your main per-sample CSV (e.g., ood_results_energy_calibrated.csv)
  --delta_csv : path to *_delta.csv with energy_before/after/delta_energy (optional)
  --out_dir   : folder to save plots and summary CSV (default: ../outputs)

Outputs:
  <out_dir>/compare_methods_bar.png
  <out_dir>/compare_methods_roc.png
  <out_dir>/methods_auroc_summary.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


# ----------------------- I/O helpers -----------------------
def load_data(main_csv, delta_csv=None):
    df = pd.read_csv(main_csv)
    if delta_csv and os.path.isfile(delta_csv):
        dfd = pd.read_csv(delta_csv)
        # align by path if present in both; else assume same order
        if "path" in df.columns and "path" in dfd.columns:
            dfd = dfd.set_index("path").reindex(df["path"]).reset_index()
        df = pd.concat([df.reset_index(drop=True), dfd.reset_index(drop=True)], axis=1)
    # ✅ drop duplicate columns created by concat (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def _to_1d_numeric(x):
    """Accept Series/DataFrame/ndarray; always return 1-D float array."""
    if hasattr(x, "values"):
        x = x.values
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.squeeze()
    return x.astype(float)


# ----------------------- scoring discovery -----------------------
def compute_scores(df):
    """
    Returns dict {method_name: (y_true, score_vector)}.
    Higher score => more fake.
    """
    if "label(fake=1)" not in df.columns:
        raise RuntimeError("CSV must include 'label(fake=1)' column.")
    y = df["label(fake=1)"].astype(float).values.ravel()

    methods = {}

    # ODIN anomaly score (prefer explicit column; otherwise 1 - conf)
    if "anomaly_score" in df.columns:
        methods["ODIN (1 - conf)"] = (y, _to_1d_numeric(df["anomaly_score"]))
    elif "odin_conf" in df.columns:
        methods["ODIN (1 - conf)"] = (y, _to_1d_numeric(1.0 - df["odin_conf"]))

    # ΔConfidence if available
    if "delta_conf" in df.columns:
        methods["ΔConfidence"] = (y, _to_1d_numeric(df["delta_conf"]))

    # Energy (plain) if present in main CSV
    if "energy" in df.columns:
        methods["Energy (plain)"] = (y, _to_1d_numeric(df["energy"]))

    # ΔEnergy (prefer direct column; else compute from before/after)
    if "delta_energy" in df.columns:
        methods["ΔEnergy"] = (y, _to_1d_numeric(df["delta_energy"]))
    elif {"energy_before", "energy_after"}.issubset(df.columns):
        methods["ΔEnergy"] = (
            y,
            _to_1d_numeric(df["energy_after"] - df["energy_before"])
        )

    # Mahalanobis distance
    if "mahalanobis" in df.columns:
        methods["Mahalanobis"] = (y, _to_1d_numeric(df["mahalanobis"]))

    # Combined score (if you stored it)
    if "combined_score" in df.columns:
        methods["Hybrid Combined"] = (y, _to_1d_numeric(df["combined_score"]))

    return methods


def aurocs_from_methods(methods):
    out = {}
    for name, (y, s) in methods.items():
        y = _to_1d_numeric(y)
        s = _to_1d_numeric(s)
        # ensure binary labels {0,1}
        if set(np.unique(y)) - {0.0, 1.0}:
            y = (y > 0).astype(float)
        out[name] = roc_auc_score(y, s)
    return out


# ----------------------- plotting -----------------------
def plot_bar(aurocs, out_png):
    names = list(aurocs.keys())
    vals = [aurocs[n] for n in names]
    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]

    plt.figure(figsize=(8.4, 4.8))
    plt.bar(names, vals)
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUROC")
    plt.title("Anomaly Detection Methods — AUROC Comparison")
    for i, v in enumerate(vals):
        plt.text(i, min(0.98, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved bar chart → {out_png}")


def plot_rocs(methods, out_png):
    plt.figure(figsize=(6.2, 6.2))
    for name, (y, s) in methods.items():
        y = _to_1d_numeric(y)
        s = _to_1d_numeric(s)
        fpr, tpr, _ = roc_curve(y, s)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Anomaly Detection")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved ROC curves → {out_png}")


# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_csv", required=True,
                    help="Path to main per-sample CSV (e.g., ../outputs/ood_results_energy_calibrated.csv)")
    ap.add_argument("--delta_csv", default=None,
                    help="Path to *_delta.csv with energy_before/after/delta_energy (optional)")
    ap.add_argument("--out_dir", default="../outputs", help="Where to save plots/summary")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(args.main_csv, args.delta_csv)
    methods = compute_scores(df)
    if not methods:
        raise RuntimeError(
            "No recognizable score columns found. Expected one or more of:\n"
            "  - anomaly_score or odin_conf\n"
            "  - delta_conf\n"
            "  - energy (plain)\n"
            "  - delta_energy or (energy_before & energy_after)\n"
            "  - mahalanobis\n"
            "  - combined_score (optional)\n"
        )

    # AUROC table (robust to Series/DataFrame)
    aurocs = aurocs_from_methods(methods)
    print("\n== AUROC (computed from CSV) ==")
    for k, v in sorted(aurocs.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{k:20s} : {v:.4f}")

    # Save summary CSV
    summary_csv = os.path.join(args.out_dir, "methods_auroc_summary.csv")
    pd.DataFrame(
        {"method": list(aurocs.keys()), "auroc": list(aurocs.values())}
    ).sort_values("auroc", ascending=False).to_csv(summary_csv, index=False)
    print(f"Saved AUROC summary → {summary_csv}")

    # Plots
    bar_png = os.path.join(args.out_dir, "compare_methods_bar.png")
    roc_png = os.path.join(args.out_dir, "compare_methods_roc.png")
    plot_bar(aurocs, bar_png)
    plot_rocs(methods, roc_png)


if __name__ == "__main__":
    main()
