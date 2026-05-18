#!/usr/bin/env python3
"""
compare_algos.py — Compare BMSSP vs Dijkstra on multiple datasets.
Reads all result_*.csv files in the current directory and plots:
1. Runtime comparison
2. Speed ratio (BMSSP / Dijkstra)
3. Which algorithm wins on each dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# --- Load all result CSVs ---
def load_all_results(pattern="result_*.csv"):
    files = glob.glob(pattern)
    if not files:
        print("❌ No result_*.csv files found in current directory.")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Normalize dataset column (in case it stores file path)
            df["dataset"] = os.path.basename(df["dataset"].iloc[0]).replace(".edgelist", "")
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Could not read {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# --- Main comparison ---
def compare_results(df):
    if df.empty:
        print("No data to plot.")
        return

    # Compute derived metrics
    df["speed_ratio"] = df["time_bmssp"] / df["time_dijkstra"]
    df["faster_algo"] = df.apply(lambda r: "BMSSP" if r["time_bmssp"] < r["time_dijkstra"] else "Dijkstra", axis=1)

    print("\n=== Summary ===")
    print(df[["dataset", "n", "m", "time_dijkstra", "time_bmssp", "speed_ratio", "faster_algo"]])
    print("\nAverage speed ratio:", df["speed_ratio"].mean())

    # --- Plot 1: Absolute runtimes ---
    plt.figure(figsize=(8, 5))
    plt.bar(df["dataset"], df["time_dijkstra"], label="Dijkstra")
    plt.bar(df["dataset"], df["time_bmssp"], label="BMSSP", alpha=0.7)
    plt.title("Runtime Comparison (seconds)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("comparison_runtime.png")
    plt.show()

    # --- Plot 2: Speed ratio ---
    plt.figure(figsize=(8, 5))
    plt.bar(df["dataset"], df["speed_ratio"], color="skyblue")
    plt.axhline(1, color="red", linestyle="--", linewidth=1)
    plt.title("BMSSP / Dijkstra Speed Ratio (lower = faster BMSSP)")
    plt.ylabel("Ratio")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("comparison_ratio.png")
    plt.show()

    # --- Plot 3: Winner chart ---
    colors = df["faster_algo"].map({"BMSSP": "green", "Dijkstra": "orange"})
    plt.figure(figsize=(8, 5))
    plt.bar(df["dataset"], [1]*len(df), color=colors)
    for i, a in enumerate(df["faster_algo"]):
        plt.text(i, 0.5, a, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    plt.title("Algorithm Winner per Dataset")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("comparison_winners.png")
    plt.show()

if __name__ == "__main__":
    df = load_all_results()
    compare_results(df)
