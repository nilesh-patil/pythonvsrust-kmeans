---
title: Parallelism
---

The honest number first: handing the assignment step to Rayon buys a best-case 1.32x at the largest workload, and on the small slices it's a regression. The rest of this page is the why — what parallelizes cleanly, what doesn't, and what would have to change to raise that ceiling.

<span class="newthought">Lloyd's hot loop</span> has two steps that are data-parallel and one that isn't worth the trouble. The assignment step maps each point to its nearest centroid, and every point's answer is independent of every other point's, so it's a clean parallel map. The update step sums each cluster's members and divides, which is a reduction. The convergence check compares the old centroids against the new ones, which is cheap and stays serial.

The assignment step is the easy win. The centroids are read-only while it runs, so Rayon workers share them with no synchronization at all — each worker just computes nearest-centroid for its slice of points and the results collect back into a label vector:

```rust
fn assign_clusters_parallel(&self, data: &[DataPoint]) -> Vec<usize> {
    let centroids = &self.centroids;
    data.par_iter()
        .map(|point| {
            let mut min_dist = f64::INFINITY;
            let mut best = 0usize;
            for (idx, centroid) in centroids.iter().enumerate() {
                let d = Self::euclidean_distance(&point.features, centroid);
                if d < min_dist { min_dist = d; best = idx; }
            }
            best
        })
        .collect()
}
```

The update step needs more care, because a naive parallel sum into shared per-cluster buffers would have every worker fighting over the same memory. The fix is the standard map-reduce shape: each Rayon worker folds its slice into a private accumulator, a `Vec<(sum, count)>` with one entry per cluster, and then a `reduce` merges the per-worker accumulators by element-wise addition. No worker ever writes to another's buffer, so there's no contention and no lock.

```rust
let merged = data
    .par_iter()
    .zip(labels.par_iter())
    .fold(zero_acc, |mut acc, (point, &label)| {
        let (ref mut sum, ref mut count) = acc[label];
        for (s, &v) in sum.iter_mut().zip(point.features.iter()) { *s += v; }
        *count += 1;
        acc
    })
    .reduce(zero_acc, |mut a, b| { /* element-wise merge of per-thread sums */ a });
```

<figure class="figure-wide">
  <img src="{{ '/assets/images/diagrams/rayon-parallel.svg' | relative_url }}" alt="Diagram of the Rayon work split: par_iter fans the data rows across worker threads for the embarrassingly parallel argmin assignment, then each thread's private sum-and-count accumulators fold and reduce into the new centroids at a single synchronization point.">
  <figcaption>What Rayon actually parallelises. <code>par_iter()</code> fans the rows across workers for the argmin assignment, which is embarrassingly parallel; then each thread's private (sum, count) accumulators fold and reduce into the new centroids — the one synchronization point.</figcaption>
</figure>

So assignment is a map, update is a reduce, and the empty-cluster handling (reseed to a random point) matches the serial path exactly, which is what keeps the parallel build's clustering quality bit-identical to serial Rust on every paired row.

## What it actually bought

<figure>
  <img src="{{ '/assets/images/parallel_scaling.svg' | relative_url }}" alt="Parallel speedup versus thread count across sample sizes, rising toward 1.3x only at the largest workloads.">
  <figcaption>Serial-relative speedup from the Rust thread sweep on fresh n × 32-feature datasets at k_max=32. The gain is noise or worse on small slices and tops out around 1.3x at the largest.</figcaption>
</figure>

The thread sweep runs the same binary across the suite's log2 sample sequence, 1k through 256k rows, and records each slice alongside the main results. The picture is consistent and modest:

| Samples | Serial runtime | Best threads | Best parallel runtime | Best speedup | 8-thread speedup |
|--------:|---------------:|-------------:|----------------------:|-------------:|-----------------:|
| 1k | 7.27 ms | 1 | 6.67 ms | 1.09x | 0.93x |
| 2k | 11.11 ms | 4 | 9.86 ms | 1.13x | 1.02x |
| 4k | 17.32 ms | 4 | 16.46 ms | 1.05x | 0.93x |
| 8k | 33.13 ms | 2 | 30.16 ms | 1.10x | 1.04x |
| 16k | 59.00 ms | 2 | 51.83 ms | 1.14x | 1.10x |
| 32k | 109.26 ms | 8 | 95.27 ms | 1.15x | 1.15x |
| 64k | 241.72 ms | 4 | 187.97 ms | 1.29x | 1.28x |
| 128k | 425.25 ms | 4 | 360.78 ms | 1.18x | 1.15x |
| 256k | 945.28 ms | 8 | 717.42 ms | 1.32x | 1.32x |

The best speedup I observe anywhere is 1.32x, on the 256k-row slice. On the 1k and 4k slices eight threads are slower than serial, because by then the parallel machinery costs more than the work it's splitting up. The useful range is the large end of the grid, and even there 1.3x is a long way from the core count.

## Why the ceiling is so low

The first reason is structural and unfixable: Lloyd's is serial across iterations. Each iteration's centroids depend on the previous iteration's labels, so the parallelism lives entirely *within* an iteration. You can split the assignment over cores, but you cannot start iteration \\(i+1\\) before iteration \\(i\\) finishes, and there's no thread-count that buys you around that.

The second reason is the data layout, and this one I could fix. Each row is a `DataPoint { id: String, features: Vec<f64> }`, so the dataset is a vector of structs, each owning a separately heap-allocated feature vector and a string. Walking it means chasing pointers, and the distance kernel spends as much time waiting on cache misses as computing. A flat `Vec<f64>` of length \\(n \times d\\) would lay every feature out contiguously, let the prefetcher do its job, and give each Rayon worker a clean cache-friendly chunk. It's a real refactor, not a one-liner, so it hasn't landed yet.

The third reason is the benchmark itself. The CLI fits every \\(k\\) from 1 to k_max in one run, and most of those fits are small. Rayon's per-fit thread setup is fixed overhead, and on a pile of tiny fits that overhead dominates whatever the parallel assignment saves. A grid that distinguishes one large \\(k = k_{\max}\\) fit from many small ones would let the parallel path show its work instead of drowning it in startup cost.

Those last two are the changes that would move the crossover down: flatten the matrix so the assignment kernel is memory-bound on contiguous data, and benchmark a single large fit rather than a sweep of small ones. Both are on the list. Until then the parallel binary stays an opt-in experiment: it's correct and it's a clean piece of Rayon, it just doesn't yet earn its place as the default.
