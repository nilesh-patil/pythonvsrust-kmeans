# Diagram manifest

Hand-built SVGs for the Tufte editorial redesign. All four use the brief's
tokens (paper `#fffff8`, ink `#151515`, faint `#595959`, rules `#dcdcd4`,
accent `#a32015`; Python `#3d6b9e`, scikit-learn `#c98c1f`, Rust `#b7410e`,
Rust-Parallel `#7a2e0c`), Newsreader serif text, hairline strokes, no shadows,
no gradients. Each is `viewBox`-only (responsive) and legible at the 720px
column width. Captions below are in the author's first-person engineer voice.

| File | Suggested figcaption | Page / section |
|---|---|---|
| `lloyds-iteration.svg` | One pass of Lloyd's algorithm: I assign each point to its nearest centroid, move every centroid to its members' mean, and repeat until the labels stop moving. | **algorithms.md** — alongside the "Algorithm in four lines" / Lloyd's description (top of the page, after the opening definition). |
| `kmeanspp-seeding.svg` | Uniform random seeding can drop both centroids in one blob and split it while a real cluster goes unseeded; sampling each next seed proportional to D² jumps to the far blob instead. | **algorithms.md** — in the "Random init vs k-means++" section, replacing or supporting the `init_comparison.png` discussion. |
| `rayon-parallel.svg` | What Rayon actually parallelises: `par_iter()` fans the rows across workers for the argmin assignment (embarrassingly parallel), then each thread's private (sum, count) accumulators fold and reduce into the new centroids — the one synchronization point. | **parallel.md** — under "How", next to the `par_iter()`/`fold`/`reduce` code block. |
| `workload-anatomy.svg` | What the benchmark times is the whole CLI process — launch, CSV read, every `k = 1..k_max` fit, then the CSV write — not the clustering kernel in isolation, which is why small-`k` overhead dilutes the parallel speedup. | **benchmarks.md** (preferred) or **index.md** — wherever the end-to-end timing methodology / caveat is introduced. |

## Notes for the integration pass

- The writer's placeholders use `<!-- DIAGRAM: <topic> -->`; wire each file into
  a `<figure> … <figcaption>` using the captions above.
- `rayon-parallel.svg` is the widest (960×520 viewBox) and reads best as a
  `.figure-wide` break-out. `lloyds-iteration.svg` (960×360) and
  `workload-anatomy.svg` (960×332) also benefit from `.figure-wide`.
  `kmeanspp-seeding.svg` (960×380) is fine at column width but wide too.
- All four match the prose claims as verified against
  `src/rust_impl/src/lib.rs` and `main.rs` (see report).
