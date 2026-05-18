# Graph-Aware Recommendation System using Batch Multi-Source Shortest Path (BMSSP) and Machine Learning

**B.Tech Final Year Project | 8th Semester**  
**International Institute of Information Technology, Bhubaneswar**  
**Department of Computer Science and Engineering**

| | |
|---|---|
| **Authors** | Shreyashi Panigrahy (B122110) · Sipra Mohanty (B122112) |
| **Guide** | Dr. Tushar Ranjan Sahoo |
| **HOD** | Prof. Ajay Kumar Dash |

---

## Overview

This project implements the **Batch Multi-Source Shortest Path (BMSSP)** algorithm in C++ and applies it to build a **graph-aware recommendation system**. User–item interactions are modelled as a weighted bipartite graph; BMSSP extracts structural proximity features (shortest-path distances, reachability) that are combined with collaborative filtering statistics and fed into a **HistGradientBoosting** classifier trained on the **MovieLens 1M** dataset.

The core hypothesis: *BMSSP-derived graph-proximity features, when combined with collaborative filtering signals, produce higher ranking quality (NDCG@K) than either source alone.*

---

## Repository Structure

```
project/
├── README.md
├── overleaf.tex                        # Full LaTeX report
├── references.bib                      # BibTeX bibliography
├── IIIT_Bhubaneswar_Logo-removebg-preview.png
│
├── part1_bmssp_benchmark/              # Part 1 — BMSSP algorithm benchmarking
│   ├── final.cpp                       # Core BMSSP implementation
│   ├── benchmark.cpp                   # Benchmark on polblogs
│   ├── benchmark_stanf.cpp             # Benchmark on web-Stanford
│   ├── benchmark_web.cpp               # Benchmark on web-NotreDame
│   ├── benchmark_snap.cpp              # Benchmark on wiki-Vote
│   ├── convert_snap_to_edgelist.py     # SNAP format → edgelist
│   ├── convert_mtx_to_edgelist.py      # MTX format → edgelist
│   ├── compare_algos.py                # Runtime comparison plots
│   ├── plot_summary.py                 # Summary visualisation
│   ├── combine.py                      # Merge result CSVs
│   └── summary_results.csv             # Measured benchmark results
│
└── part2_recommendation/               # Part 2 — BMSSP + ML pipeline
    ├── src/
    │   ├── bmssp_features.cpp          # C++ BMSSP feature extractor
    │   ├── ml_pipeline.py              # Training, evaluation, ranking metrics
    │   ├── data_loader.py              # MovieLens data loading utilities
    │   ├── graph_builder.py            # Bipartite graph construction
    │   ├── features.py                 # Feature extraction helpers
    │   ├── model.py                    # Model definitions and metrics
    │   ├── recommend.py                # Recommendation generation
    │   └── generate_advanced_plots.py  # Evaluation plot generation
    ├── data/                           # Place ml-1m/ dataset here
    ├── results/
    │   └── recommendation_results.csv
    └── plots/                          # All evaluation figures (10 PNGs)
        ├── 1_roc_auc_curve.png
        ├── 2_precision_recall_curve.png
        ├── 3_ranking_metrics.png
        ├── 4_hit_rate.png
        ├── 5_feature_importance.png
        ├── 6_score_distribution.png
        ├── 7_cumulative_gain.png
        ├── 8_bmssp_distance_vs_quality.png
        ├── 9_cold_start_analysis.png
        └── 10_model_comparison_roc.png
```

---

## Part 1 — BMSSP Algorithm Benchmarking

### Goal
Benchmark the BMSSP algorithm against Dijkstra's single-source shortest-path algorithm on four real-world graphs of varying size and density.

### Benchmark Datasets

| Dataset | Nodes | Edges | Description |
|---------|------:|------:|-------------|
| polblogs | 1,490 | 19,022 | Political blogs network |
| wiki-Vote | 8,298 | 103,689 | Wikipedia voting network |
| web-Stanford | 281,904 | 2,312,497 | Stanford web crawl graph |
| web-NotreDame | 325,729 | 1,497,134 | Notre Dame web crawl graph |

### How to Run

```bash
cd part1_bmssp_benchmark/

# Compile and run (example: polblogs)
g++ -std=c++17 -O3 -o benchmark benchmark.cpp
./benchmark

# Generate comparison plots
python3 compare_algos.py
python3 plot_summary.py
```

### Measured Runtime Results

| Dataset | Dijkstra (ms) | BMSSP (ms) | Speedup |
|---------|-------------:|----------:|--------:|
| polblogs | 0.122 | 0.087 | ~1.4× |
| web-NotreDame | 43.66 | 0.616 | **~70×** |
| web-Stanford | 0.272 | 0.408 | 0.67× |
| wiki-Vote | 0.014 | 0.025 | 0.56× |

> BMSSP achieves significant speedup on large, dense graphs (web-NotreDame). On smaller or sparser graphs, the batch-initialisation overhead can make it comparable to or slower than a single Dijkstra call — consistent with the theoretical expectation.

---

## Part 2 — BMSSP + ML Recommendation System

### Goal
Build a graph-aware recommendation pipeline using BMSSP-derived structural features combined with collaborative filtering statistics, evaluated on the MovieLens 1M benchmark.

### Dataset — MovieLens 1M

| Statistic | Value |
|-----------|-------|
| Users | 6,040 |
| Movies | 3,952 |
| Ratings | 1,000,209 |
| Rating scale | 1–5 stars |
| Density | ~4.17% |

Download from: <https://grouplens.org/datasets/movielens/1m/>  
Place in `part2_recommendation/data/ml-1m/`.

### Methodology

#### 1. Preprocessing
- Ratings ≥ 4 → positive label (1); rating = 3 → negative label (0); < 3 excluded.
- Leave-one-out evaluation: one positive item per user held out for testing.
- Negative sampling: 4 random unobserved items per positive training pair; 99 per test case.

#### 2. Graph Construction
- Bipartite graph G = (U ∪ I, E) with binary edge weights (w = 1 for ratings ≥ 4).
- Adjacency-list representation in C++ for memory efficiency.

#### 3. Feature Extraction — 15 Features per (user, item) Pair

**BMSSP Structural Features (5)**

| Feature | Description |
|---------|-------------|
| `weighted_distance` | Shortest-path distance from user to candidate item |
| `min_wdist_liked` | Minimum distance from candidate to any liked item |
| `avg_wdist_liked` | Average distance to liked items |
| `inverse_wdist` | Inverse distance (higher = closer) |
| `liked_reachability` | Fraction of liked items reachable within k hops |

**Graph Topology Features (3)**

| Feature | Description |
|---------|-------------|
| `2_hop_reachable` | Whether item is reachable in 2 hops |
| `user_degree` | Number of items connected to the user |
| `movie_degree` | Number of users connected to the item |

**Collaborative Filtering Features (7)**

| Feature | Description |
|---------|-------------|
| `user_avg_rating` | Mean rating given by user |
| `movie_avg_rating` | Mean rating received by item |
| `baseline_prediction` | user_avg + movie_avg − global_avg |
| `user_count` | Total ratings by user |
| `movie_count` | Total ratings for item |
| `user_std_rating` | Variance of user ratings |
| `movie_std_rating` | Variance of item ratings |

#### 4. Model
- **Algorithm**: `HistGradientBoostingClassifier` (scikit-learn)
- **Hyperparameters**: `max_iter=500`, `max_depth=8`, `learning_rate=0.08`, `min_samples_leaf=20`, `l2_regularization=0.05`
- **Objective**: Cross-entropy surrogate for NDCG@K maximisation
- **Training time**: ~12 seconds on 900K+ samples

#### 5. Evaluation
- Primary metrics: Hit@K, NDCG@K, Precision@K (ranking quality)
- Secondary metrics: AUC-ROC, Accuracy (classification quality)
- Evaluated over 1,762 users with at least one positive test item

### How to Run

```bash
cd part2_recommendation/src/

# Step 1: Compile C++ BMSSP feature extractor
g++ -std=c++17 -O3 -o bmssp_features bmssp_features.cpp

# Step 2: Run BMSSP to extract features (~30s)
./bmssp_features

# Step 3: Train model and evaluate (~30s)
python3 ml_pipeline.py

# Step 4: Generate evaluation plots
python3 generate_advanced_plots.py
```

### Results

#### Primary Ranking Metrics

| Metric | Value |
|--------|------:|
| **Hit@20** | **92.22%** |
| **Hit@10** | **81.44%** |
| **Hit@5** | **67.71%** |
| NDCG@5 | 0.2825 |
| NDCG@10 | 0.2662 |
| NDCG@20 | 0.2467 |
| Precision@5 | 26.11% |
| Precision@10 | 22.65% |
| Precision@20 | 17.75% |

#### Secondary Classification Metrics

| Metric | Value |
|--------|------:|
| AUC-ROC | 80.31% |
| Accuracy | 72.39% |

> **Note**: Accuracy (72.4%) is not the same as Hit@20 (92.2%). Accuracy counts correct per-pair binary predictions; Hit@20 asks whether at least one relevant item appears in the top-20 list — the more meaningful measure for real recommendation scenarios.

#### Baseline Comparison

| Method | Hit@10 | Hit@20 | NDCG@5 | AUC-ROC |
|--------|-------:|-------:|-------:|--------:|
| Random | 8.2% | 15.7% | 2.1% | 49.8% |
| Popularity | 42.3% | 58.9% | 11.2% | 62.7% |
| Collaborative Filtering (SVD) | 67.8% | 79.1% | 19.7% | 74.2% |
| Single-Source Graph | 78.9% | 89.4% | 26.1% | 78.9% |
| **BMSSP + ML (ours)** | **81.4%** | **92.2%** | **28.3%** | **80.3%** |

---

## Requirements

### C++
- C++17 compiler: `g++` ≥ 9.3 or `clang++` ≥ 10
- Compile flag: `-O3` recommended

### Python
- Python 3.8+

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## Optimization Formulation

The system maximises NDCG@K as the primary objective:

```
max_θ  (1/|U|) Σ_{u∈U} NDCG@K(u, f_θ(X_u))
```

Since NDCG@K is non-differentiable, training minimises a cross-entropy surrogate with ℓ₂ regularisation:

```
min_θ  (1/N) Σ_{(u,i)} ℓ_CE(rel(u,i), f_θ(x_{u,i})) + λ‖θ‖₂²
```

where `x_{u,i}` is the 15-dimensional BMSSP + collaborative feature vector and `f_θ` is the HistGradientBoosting model.

---

## References

1. Adomavicius, G. & Tuzhilin, A. — "Toward the Next Generation of Recommender Systems" (IEEE TKDE, 2005)
2. Koren, Y., Bell, R. & Volinsky, C. — "Matrix Factorization Techniques for Recommender Systems" (IEEE Computer, 2009)
3. Kipf, T.N. & Welling, M. — "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR, 2017)
4. Wang, X. et al. — "Neural Graph Collaborative Filtering" (SIGIR, 2019)
5. Hamilton, W., Ying, R. & Leskovec, J. — "Inductive Representation Learning on Large Graphs" (NeurIPS, 2017)
6. He, X. et al. — "Neural Collaborative Filtering" (WWW, 2017)
7. Harper, F.M. & Konstan, J.A. — "The MovieLens Datasets: History and Context" (ACM TiiS, 2015)
8. Dijkstra, E.W. — "A Note on Two Problems in Connexion with Graphs" (Numerische Mathematik, 1959)
