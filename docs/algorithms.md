---
title: Algorithms
---

# Lloyd's algorithm and the cost of bad initialization

K-Means is **Lloyd's algorithm**: alternate between assigning each point to its nearest centroid and updating each centroid to the mean of its members. Iterate until labels stop changing.

The catch: the final result depends on where you start. The two extremes:

| Init scheme | What it does | When it bites you |
|---|---|---|
| **Random** | Pick `k` data points uniformly at random | Two centroids often land in the same dense blob, leaving a real cluster unseeded |
| **k-means++** | First centroid uniform; each next centroid sampled proportional to D²(point, nearest existing centroid) | Almost always picks one centroid per region, even on adversarial seeds |

## Animation: random vs k-means++ vs pathological

| random init | k-means++ init | pathological random seed |
|---|---|---|
| ![random](assets/animations/convergence_random.gif) | ![kmeans++](assets/animations/convergence_kpp.gif) | ![pathological](assets/animations/convergence_pathological.gif) |

Notice how the pathological random seed (rightmost) leaves two centroids stuck in the same blob for many iterations. k-means++ (middle) converges in 2–3 iterations because each blob is hit on the first pass.

## Inertia: random vs k-means++

Across three dataset sizes (small/medium/large), k-means++ produces **37–54 % lower inertia** averaged over 10 seeds:

![init comparison](assets/images/init_comparison.png)

## Algorithm summary

For a dataset `X ∈ ℝ^{n×d}` and target cluster count `k`:

1. **Initialize** `k` centroids (random or k-means++).
2. **Assign**: every point goes to its nearest centroid (Euclidean).
3. **Update**: every centroid becomes the mean of its assigned points.
4. **Check** convergence (labels stable) — break, else loop.

The pure-Python implementation in [`src/python_impl/kmeans.py`](https://github.com/nilesh-patil/pythonvsrust-kmeans/blob/master/src/python_impl/kmeans.py) is ~150 lines of NumPy and serves as the readable reference. The Rust port in [`src/rust_impl/`](https://github.com/nilesh-patil/pythonvsrust-kmeans/tree/master/src/rust_impl) is a faithful, single-threaded port with the same algorithm.
