#!/bin/bash
# Run the complete BMSSP + ML Recommendation Pipeline

echo "=== BMSSP + ML Graph-Aware Recommendation System ==="
echo ""

# Check data
if [ ! -f "data/ml-1m/ratings.dat" ]; then
    echo "ERROR: MovieLens 1M data not found!"
    echo "Download from: https://grouplens.org/datasets/movielens/1m/"
    echo "Place ml-1m/ folder inside data/"
    exit 1
fi

# Step 1: Compile and run C++ BMSSP
echo "[Phase 1] Compiling BMSSP feature extractor..."
cd src/
g++ -std=c++17 -O2 -o bmssp_features bmssp_features.cpp
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "[Phase 1] Running BMSSP feature extraction..."
./bmssp_features
if [ $? -ne 0 ]; then
    echo "Feature extraction failed!"
    exit 1
fi

# Step 2: Run ML pipeline
echo ""
echo "[Phase 2] Running ML pipeline..."
python3 ml_pipeline.py

echo ""
echo "=== Done! Check results/ and plots/ ==="
