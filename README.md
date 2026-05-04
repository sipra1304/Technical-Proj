# Graph-Aware Recommendation System using BMSSP & ML

**BTech Project | 8th Semester**  
**Authors:** Sipra & Shreyashi

---

## Overview

This project implements a **Graph-Aware Recommendation System** that applies the **Batch Multi-Source Shortest Path (BMSSP)** algorithm for structural feature extraction in a recommendation framework. The system models user–item interactions as a weighted bipartite graph and leverages shortest-path distances as features for machine learning-based recommendation prediction.

---

## Project Structure

```
project/
├── README.md
├── part1_bmssp_benchmark/          # BMSSP Algorithm Benchmarking
│   ├── final.cpp                   # BMSSP implementation (paper-faithful)
│   ├── benchmark.cpp               # Benchmark on polblogs dataset
│   ├── benchmark_stanf.cpp         # Benchmark on web-Stanford dataset
│   ├── benchmark_web.cpp           # Benchmark on web-NotreDame dataset
│   ├── benchmark_snap.cpp          # Benchmark on wiki-Vote dataset
│   ├── convert_snap_to_edgelist.py # SNAP format → edgelist converter
│   ├── convert_mtx_to_edgelist.py  # MTX format → edgelist converter
│   ├── compare_algos.py            # Comparison plots
│   ├── plot_summary.py             # Summary visualization
│   ├── combine.py                  # Combine result CSVs
│   └── summary_results.csv         # Benchmark results
│
└── part2_recommendation/           # BMSSP + ML Recommendation System
    ├── src/
    │   ├── bmssp_features.cpp      # C++ BMSSP feature extraction
    │   ├── ml_pipeline.py          # ML model training & evaluation
    │   ├── data_loader.py          # MovieLens data loading utilities
    │   ├── graph_builder.py        # Bipartite graph construction + BMSSP
    │   ├── features.py             # Feature extraction functions
    │   ├── model.py                # ML model definitions & metrics
    │   └── recommend.py            # Recommendation generation
    ├── data/                        # Place ml-1m/ dataset here
    ├── results/
    │   └── recommendation_results.csv
    └── plots/
        ├── ranking_metrics.png
        └── hit_rate.png
```

---

## Part 1: BMSSP Algorithm Benchmarking

### Objective
Implement the BMSSP algorithm from "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths" (Duan et al.) and benchmark it against Dijkstra's algorithm on real-world graph datasets.

### Datasets Used
| Dataset | Nodes | Edges | Source |
|---------|-------|-------|--------|
| polblogs | 1,490 | 19,022 | Political blogs network |
| wiki-Vote | 8,298 | 103,689 | Wikipedia voting network |
| web-Stanford | 281,904 | 2,312,497 | Stanford web graph |
| web-NotreDame | 325,729 | 1,497,134 | Notre Dame web graph |

### How to Run
```bash
cd part1_bmssp_benchmark/
g++ -std=c++17 -O2 -o benchmark benchmark.cpp
./benchmark
python3 compare_algos.py
```

### Results
| Dataset | Dijkstra (s) | BMSSP (s) |
|---------|-------------|-----------|
| polblogs | 0.000122 | 0.000087 |
| web-NotreDame | 0.043659 | 0.000616 |
| web-Stanford | 0.000272 | 0.000408 |
| wiki-Vote | 0.000014 | 0.000025 |

---

## Part 2: BMSSP + ML Recommendation System

### Objective
Design a graph-aware recommendation framework using structural relationships in user–item data, with BMSSP for multi-source distance computation.

### Dataset
**MovieLens 1M** — Public benchmark dataset (GroupLens Research Lab)
- 6,040 users
- 3,952 movies
- 1,000,209 ratings (scale: 1–5)

### Methodology

#### 1. Graph Construction
- Bipartite graph: users ↔ movies
- Edge weight = (6 − rating): higher ratings create shorter paths
- Adjacency list representation for memory efficiency

#### 2. BMSSP Implementation
- Weighted Dijkstra-based BMSSP for single-source shortest paths
- Computes distances from each user to all reachable movies
- Leave-one-out evaluation for training (removes direct edge to prevent leakage)

#### 3. Structural Feature Extraction (15 features)

**BMSSP Structural Features (6):**
- `weighted_distance`: Shortest path distance from user to candidate movie
- `min_wdist_liked`: Minimum distance from candidate to user's liked movies
- `avg_wdist_liked`: Average distance to liked movies
- `inverse_wdist`: Inverse distance score (higher = closer)
- `num_liked_close_norm`: Fraction of liked movies within distance threshold
- `liked_reachability`: Fraction of liked movies reachable in graph

**Graph Topology Features (2):**
- `user_degree`: Number of movies connected to user
- `movie_degree`: Number of users connected to movie

**Collaborative Features (7):**
- `user_avg_rating`, `movie_avg_rating`: Mean ratings
- `baseline_prediction`: user_avg + movie_avg − global_avg
- `user_count`, `movie_count`: Activity counts
- `user_std_rating`, `movie_std_rating`: Rating variance

#### 4. ML Model
- **Algorithm:** HistGradientBoosting (sklearn)
- 500 iterations, max depth 8, learning rate 0.08
- Trained on 900K+ samples in ~12 seconds

#### 5. Evaluation Protocol
Standard recommendation evaluation with negative sampling:
- For each test user: rank positive items (rating ≥ 4) against 49 random unrated movies
- Metrics: Hit@K, Precision@K, Recall@K, NDCG@K

### How to Run
```bash
cd part2_recommendation/

# Download MovieLens 1M (place in data/ml-1m/)
# https://grouplens.org/datasets/movielens/1m/

# Step 1: Compile and run C++ BMSSP feature extraction (~30s)
cd src/
g++ -std=c++17 -O2 -o bmssp_features bmssp_features.cpp
./bmssp_features

# Step 2: Run ML pipeline (~30s)
python3 ml_pipeline.py
```

### Results

| Metric | Value |
|--------|-------|
| **Hit@20** | **92.2%** |
| **Hit@10** | **81.4%** |
| **Hit@5** | **67.7%** |
| **AUC-ROC** | **80.3%** |
| Accuracy | 72.4% |
| NDCG@5 | 28.3% |
| NDCG@10 | 26.6% |
| NDCG@20 | 24.7% |
| Precision@5 | 26.1% |
| Precision@10 | 22.7% |
| Precision@20 | 17.8% |

---

## Requirements

### System
- C++17 compiler (g++ / clang++)
- Python 3.8+

### Python Packages
```
numpy
pandas
scikit-learn
matplotlib
```

Install: `pip install numpy pandas scikit-learn matplotlib`

---

## References

1. Duan, R., Mao, Y., Shu, X., Yin, Q. — "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"
2. Harper, F.M. & Konstan, J.A. — "The MovieLens Datasets: History and Context" (ACM TiiS, 2015)
3. He, X. et al. — "Neural Collaborative Filtering" (WWW 2017)
