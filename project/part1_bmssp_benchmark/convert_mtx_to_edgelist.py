#!/usr/bin/env python3
# convert_mtx_to_edgelist.py
import sys, random

if len(sys.argv) < 3:
    print("Usage: python3 convert_mtx_to_edgelist.py polblogs.mtx output.edgelist [seed]")
    sys.exit(1)

inp = sys.argv[1]
out = sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
random.seed(seed)

edges = []
n = m = 0

with open(inp) as f:
    for line in f:
        if line.startswith('%') or line.strip() == '':
            continue
        parts = line.split()
        if len(parts) == 3 and n == 0:
            # header line: nrows ncols nnz
            n = int(parts[0])
            m = int(parts[2])
            continue
        if len(parts) >= 2:
            u = int(parts[0]) - 1  # convert to 0-based indexing
            v = int(parts[1]) - 1
            if u != v:
                w = 1.0 + random.random() * 9.0
                edges.append((u, v, w))

with open(out, 'w') as fo:
    fo.write(f"{n} {len(edges)}\n")
    for u, v, w in edges:
        fo.write(f"{u} {v} {w:.6f}\n")

print(f"Converted {len(edges)} edges, {n} vertices → {out}")
