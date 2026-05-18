#!/usr/bin/env python3
import sys, random

if len(sys.argv) < 3:
    print("Usage: python3 convert_snap_to_edgelist.py INPUT.txt OUTPUT.edgelist [seed]")
    sys.exit(1)

inp, out = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
random.seed(seed)

edges = []
n = 0
with open(inp, 'r') as f:
    for line in f:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.strip().split()
        if len(parts) < 2: 
            continue
        u, v = int(parts[0]), int(parts[1])
        w = 1.0 + random.random() * 9.0
        edges.append((u, v, w))
        n = max(n, u, v)

n += 1
print(f"Detected n={n}, edges={len(edges)}")
with open(out, 'w') as fo:
    fo.write(f"{n} {len(edges)}\n")
    for (u, v, w) in edges:
        fo.write(f"{u} {v} {w:.6f}\n")
print(f"Written weighted graph to {out}")
