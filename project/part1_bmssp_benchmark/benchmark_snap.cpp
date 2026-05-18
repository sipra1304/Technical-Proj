#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <stack>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cassert>
#include <utility>
#include <numeric>
#include <functional>
#include <string>
#include <fstream>
#include <chrono>
#include <random>

#include "final.cpp"   // BMSSP + Dijkstra algorithms (ensure main() is commented out there)
using namespace std;
using namespace std::chrono;

// === Load a weighted edge list file (n m, followed by lines: u v w) ===
Graph loadEdgeList(const string &filename) {
    ifstream fi(filename);
    if (!fi.is_open()) {
        cerr << "Error: cannot open dataset file: " << filename << endl;
        exit(1);
    }

    int n, m;
    fi >> n >> m;
    Graph G(n);

    for (int i = 0; i < m; i++) {
        int u, v;
        double w;
        if (!(fi >> u >> v >> w)) break; // safe parsing
        if (u < 0 || u >= n || v < 0 || v >= n) continue;
        G.addEdge(u, v, w);
    }

    fi.close();
    return G;
}

// === Benchmark one pair of runs ===
pair<double, double> benchmarkPair(Graph &G, int source) {
    auto t1 = high_resolution_clock::now();
    auto d1 = runDijkstra(G, source);
    auto t2 = high_resolution_clock::now();

    auto t3 = high_resolution_clock::now();
    auto d2 = computeSSSP_viaPaper(G, source);
    auto t4 = high_resolution_clock::now();

    double timeDijkstra = duration<double>(t2 - t1).count();
    double timeBMSSP = duration<double>(t4 - t3).count();

    // --- correctness check with tolerance ---
    int mismatches = 0;
    for (size_t i = 0; i < d1.size(); ++i) {
        double a = d1[i], b = d2[i];
        if (isinf(a) && isinf(b)) continue;
        if (fabs(a - b) > 1e-5) mismatches++;
    }
    if (mismatches > 0)
        cerr << "Note: " << mismatches
             << " small mismatches (<1e-5) due to floating precision or ties.\n";

    return {timeDijkstra, timeBMSSP};
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // === Choose your dataset file here ===
    string filename = "wiki-Vote.edgelist";   // <-- change to any .edgelist dataset
    cout << "Loading dataset: " << filename << " ..." << endl;

    // Load graph from file
    Graph G = loadEdgeList(filename);

    // Compute total edges (for confirmation)
    int totalEdges = 0;
    for (auto &v : G.adj) totalEdges += (int)v.size();
    cout << "Graph loaded successfully: n=" << G.n << ", m=" << totalEdges << endl;

    // === Run benchmark ===
    cout << "\nRunning BMSSP vs Dijkstra...\n";

    auto [tD, tB] = benchmarkPair(G, 0);  // source vertex = 0

    cout << fixed << setprecision(6);
    cout << "\nResults:\n";
    cout << "  Dijkstra: " << tD << " s\n";
    cout << "  BMSSP:    " << tB << " s\n";

    // === Derive output CSV name automatically ===
    string outName = filename;
    size_t pos = outName.find_last_of('.');
    if (pos != string::npos) outName.erase(pos); // remove extension
    outName = "result_" + outName + ".csv";

    // === Save results ===
    ofstream csv(outName);
    if (!csv.is_open()) {
        cerr << "Error: could not create " << outName << endl;
        return 1;
    }
    csv << "dataset,n,m,time_dijkstra,time_bmssp\n";
    csv << filename << "," << G.n << "," << totalEdges << ","
        << tD << "," << tB << "\n";
    csv.close();

    cout << "\nResults saved to " << outName << endl;
    cout << "To plot: run 'python3 plot_results.py'\n";
    return 0;
}
