#!/usr/bin/env python3
"""
combine_results.py
Combine all per-dataset result_*.csv files into a single summary_results.csv
"""
import pandas as pd
import glob

# Find all CSVs that start with result_ and end with .csv
files = sorted(glob.glob("result_*.csv"))

if not files:
    print("No result_*.csv files found!")
    exit(1)

dfs = []
for f in files:
    print(f"Reading {f}")
    df = pd.read_csv(f)
    dfs.append(df)

# Combine them vertically
summary = pd.concat(dfs, ignore_index=True)

# Save combined file
summary.to_csv("summary_results.csv", index=False)
print("\n✅ Combined results saved to summary_results.csv")
