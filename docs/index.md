---
title: Overview
---

<div class="epigraph" markdown="1">
I wrote K-Means four times (pure-NumPy Python, hand-rolled serial Rust, the same Rust with a Rayon parallel path, and a thin wrapper over scikit-learn) and benchmarked all four as end-to-end command-line jobs on my laptop. Across 648 paired runs the serial Rust port clusters about as well as the others while running roughly four to five times faster than the Python reference and using a tenth of its memory. scikit-learn is the fastest thing here only at the very largest workload, and it pays for that speed in RAM. The Rayon parallel path tops out at a 1.32x speedup, and only on the biggest jobs.
</div>

<span class="newthought">The question</span> that started this was narrow and a little smug: how much do you actually gain by porting a textbook K-Means from Python to Rust? I had a guess. What I didn't have was a measurement I trusted, so I built one. Four implementations, one shared workload, and a benchmark harness that runs each implementation as a real subprocess from the command line rather than timing a function call in a warm interpreter. That choice matters, and I'll come back to it, because it's the difference between a microbenchmark and something closer to what you'd feel using the tool.

The four paths are deliberately matched. The pure-Python implementation is about 150 lines of NumPy and exists to be readable. The Rust port follows the same Lloyd's algorithm, line for line where it can. The parallel build is that same Rust binary with `--parallel --threads 0`, so the only thing that changes is whether the assignment and update steps fan out across cores. scikit-learn stands in for "the production answer," its clustering kernels compiled C underneath. Every run on every implementation uses k-means++ seeding with a single start, including scikit-learn at `n_init=1`, which keeps the comparison about implementation mechanics rather than restart luck.<label for="sn-init" class="sidenote-toggle">⊕</label><input type="checkbox" id="sn-init" class="sidenote-state"><span class="sidenote">scikit-learn defaults to ten restarts and keeps the best. Holding it to one start is the fair comparison here but understates what it does out of the box; a "best of N" policy would be a different study.</span>

What I measured on each run is wall-clock time around the whole subprocess, sampled process memory polled every 10 ms while it runs, child-process CPU time, and the clustering quality of the result against the ground-truth labels I generated the data from. The grid is a log2 doubling sequence from 1,000 to 256,000 rows, crossed with 2, 8, and 32 features and a k-sweep up to 8 or 32 clusters, three repeats each. The canonical numbers below all come from a single benchmark run, by way of an [auditing step]({{ '/benchmarks/' | relative_url }}) that is the only place I let myself quote figures from.

## Runtime: the port pays off, but read the workload

<figure class="figure-wide">
  <img src="{{ '/assets/images/benchmark_plots_20260609_112255.svg' | relative_url }}" alt="Four-panel benchmark plot: runtime, throughput, sampled memory, and clustering quality across sample sizes for Python, scikit-learn, Rust, and Rust-Parallel.">
  <figcaption>The two Rust lines sit nearly on top of each other — Rayon barely separates from serial — and both run well below the Python reference; scikit-learn starts expensive and only catches up at the far right, where its compiled kernels finally overtake hand-written Rust.</figcaption>
</figure>

Over the whole suite the median CLI runtime is 0.197 s for Rust-Parallel, 0.201 s for serial Rust, 0.806 s for Python, and 1.843 s for scikit-learn. The unpaired medians flatter Rust a little because the grid mixes easy and hard rows, so the number I actually trust is the paired one: matched row for matched row, serial Rust is 4.5x faster than Python at the median and Rust-Parallel is 5.1x.<label for="sn-paired" class="sidenote-toggle">⊕</label><input type="checkbox" id="sn-paired" class="sidenote-state"><span class="sidenote">The paired medians come from dividing Python's wall time by each implementation's on the same dataset, repeat, and settings, then taking the median of those ratios. Means run higher (around 6.6x) because a few large rows pull them up.</span>

<figure>
  <img src="{{ '/assets/images/speedup_curve.svg' | relative_url }}" alt="Speedup over pure Python against nominal workload, on log-log axes, for the three other implementations.">
  <figcaption>Speedup over pure Python by matched workload. The Rust paths sit well above the baseline across the grid; scikit-learn only climbs past it at the largest, heaviest fits.</figcaption>
</figure>

That advantage is not uniform across scale, and the most honest way to see it is at the top of the grid. At 256k samples by 32 features with a sweep to k=32, scikit-learn is the fastest of the four at 15.36 s, but it sits at 795 MB of sampled RSS to get there. Rust-Parallel finishes in 16.23 s at 190 MB, serial Rust in 20.76 s at 194 MB, and Python trails at 46.46 s and 924 MB. So the story isn't "Rust beats everything everywhere." It's that Rust is fast and cheap, scikit-learn is fast and hungry, and Python is neither. The crossover where scikit-learn's compiled kernels overtake hand-written Rust shows up only once the workload is big enough to amortize its startup and overhead.

Everything here is end-to-end. The wall time includes process launch, reading the CSV off disk, fitting every k from 1 to k_max, and writing all the cluster columns back out. I left it that way on purpose, because that's the cost you pay when you actually run the tool, and it's where a lot of the Python tax lives.

## Where the memory goes

<figure>
  <img src="{{ '/assets/images/memory_breakdown.svg' | relative_url }}" alt="Sampled RSS per thousand samples by implementation, with Rust lowest and scikit-learn highest.">
  <figcaption>Sampled RSS normalized per thousand rows. Rust holds well under a megabyte; scikit-learn is the heaviest by a wide margin.</figcaption>
</figure>

Normalized to a thousand samples, the median sampled RSS is 0.61 MB for Rust, 0.73 MB for Rust-Parallel, 7.41 MB for Python, and 12.63 MB for scikit-learn. Rust is the clear memory leader and it isn't close. The caveat is in how that number is taken: it's a 10 ms poll of the process's resident set, not a platform max-RSS reading, so the exact ratios are directional rather than precise.<label for="sn-rss" class="sidenote-toggle">⊕</label><input type="checkbox" id="sn-rss" class="sidenote-state"><span class="sidenote">A 10 ms sampler can miss a brief allocation spike between polls. It's good enough to rank implementations and to trust an order-of-magnitude gap; I wouldn't defend the second decimal place.</span>

I want to be careful about why Rust wins here, because it's tempting to credit a clever layout I didn't write. The Rust path stores each row as `DataPoint { id: String, features: Vec<f64> }` — a vector of owned structs, not a flat contiguous matrix. The advantage comes from sidestepping NumPy's full distance matrices and the Python object and interpreter overhead, not from a perfectly packed `n × d` buffer. There's still memory left on the table in the data layout, which becomes the recurring theme on the parallelism page.

## Quality holds up

<figure>
  <img src="{{ '/assets/images/quality_runtime_pareto.svg' | relative_url }}" alt="Quality-versus-runtime Pareto plot showing all four implementations clustering near-perfectly while differing in speed.">
  <figcaption>Quality against runtime. Under single-start k-means++ all four implementations land at the top of the quality axis; the spread that's left is speed and memory.</figcaption>
</figure>

The median ARI is 1.00 for all four implementations across the suite, which is the result I most wanted to verify before quoting any speedups: the speed numbers only mean something if every implementation is solving the same problem correctly.<label for="sn-ari" class="sidenote-toggle">⊕</label><input type="checkbox" id="sn-ari" class="sidenote-state"><span class="sidenote">ARI, the adjusted Rand index, measures agreement between two labelings corrected for chance: 1.0 is identical, 0 is what you'd expect from random labels. Defined more carefully on the algorithms page.</span> The mean ARI keeps a small gap (scikit-learn at 0.999, Python at 0.980, both Rust paths at 0.974), and that residual is occasional single-start misses, not a systematic flaw. Serial and parallel Rust share the same clustering math and produce bit-identical quality on paired rows, which is exactly what I'd want from a parallelization that's only supposed to change the schedule, not the answer.

Because every run uses k-means++ and a single start, this compares the implementations under a common policy. It does not tell you who wins a "best of ten restarts" contest, which is the regime scikit-learn is tuned for. That would be a separate ablation, and an interesting one.

## Where parallelism stops paying off

Across the balanced suite Rust-Parallel edges serial Rust by median wall time, 0.197 s against 0.201 s, which is close to a rounding error. The dedicated thread sweep, using the same binary on 32 features and k_max=32 from 1k to 256k rows, tells the real story: the best speedup I see is 1.32x at 256k rows, and on the small slices eight or more threads are an outright regression. Lloyd's iterates serially even when each iteration parallelizes, the pointer-chasing data layout fights the cache, and the CLI's many small per-k fits each pay Rayon's setup cost. Two changes would move the crossover: a genuinely flat data matrix, and a benchmark that distinguishes one large k=k_max fit from a pile of tiny ones. Until then the parallel binary is an experiment I keep around, not the default I reach for.

If you'd rather watch the algorithm than read about it, the [live demo]({{ '/demo/' | relative_url }}) runs the Rust K-Means in your browser via WebAssembly: generate two-moons data, watch a random start bisect them, then try k-means++. The full methodology, the grid, and how to reproduce every number live on the [benchmarks page]({{ '/benchmarks/' | relative_url }}).
