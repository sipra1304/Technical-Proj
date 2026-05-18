#!/usr/bin/env python3
"""
plot_summary_results.py
Plot Dijkstra vs BMSSP runtimes from summary_results.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load combined data
df = pd.read_csv("summary_results.csv")

# Sort by node count for nicer plotting
df = df.sort_values("n")

plt.figure(figsize=(7,5))
plt.loglog(df["n"], df["time_dijkstra"], marker="o", label="Dijkstra (O(m log n))")
plt.loglog(df["n"], df["time_bmssp"], marker="s", label="BMSSP (O(m log^{2/3} n))")

for i, row in df.iterrows():
    plt.text(row["n"]*1.1, row["time_bmssp"]*1.05,
             row["dataset"].replace(".edgelist",""),
             fontsize=8)

plt.xlabel("Number of vertices (n)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison: Dijkstra vs BMSSP on SNAP Datasets")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plot_summary_results.png", dpi=300)
plt.show()
