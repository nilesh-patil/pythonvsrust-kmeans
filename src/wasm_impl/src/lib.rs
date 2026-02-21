//! Browser-facing K-Means: a thin wasm-bindgen wrapper around a flat-buffer
//! Lloyd's loop. f32 across the boundary keeps the JS↔Wasm copies cheap and
//! is plenty of precision for 2-D blob visualisation.

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn kpp_init(xs: &[f32], n: usize, d: usize, k: usize, rng: &mut Pcg64Mcg) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(k * d);
    let first = rng.gen_range(0..n);
    centroids.extend_from_slice(&xs[first * d..(first + 1) * d]);

    let mut closest_d2 = vec![f32::INFINITY; n];
    for chosen in 0..k - 1 {
        let centroid = &centroids[chosen * d..(chosen + 1) * d];
        for i in 0..n {
            let p = &xs[i * d..(i + 1) * d];
            let d2 = dist2(p, centroid);
            if d2 < closest_d2[i] {
                closest_d2[i] = d2;
            }
        }
        let total: f32 = closest_d2.iter().sum();
        if total <= 0.0 {
            let fallback = rng.gen_range(0..n);
            centroids.extend_from_slice(&xs[fallback * d..(fallback + 1) * d]);
            continue;
        }
        let mut r = rng.gen::<f32>() * total;
        let mut idx = n - 1;
        for (i, &w) in closest_d2.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                idx = i;
                break;
            }
        }
        centroids.extend_from_slice(&xs[idx * d..(idx + 1) * d]);
    }
    centroids
}

fn random_init(xs: &[f32], n: usize, d: usize, k: usize, rng: &mut Pcg64Mcg) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(k * d);
    let mut chosen = Vec::with_capacity(k);
    while chosen.len() < k {
        let i = rng.gen_range(0..n);
        if !chosen.contains(&i) {
            chosen.push(i);
            centroids.extend_from_slice(&xs[i * d..(i + 1) * d]);
        }
    }
    centroids
}

#[wasm_bindgen]
pub fn kmeans_fit(
    xs: &[f32],
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    seed: u32,
    use_kpp: bool,
) -> Vec<i32> {
    if n == 0 || k == 0 || d == 0 || xs.len() != n * d {
        return vec![];
    }
    let mut rng = Pcg64Mcg::seed_from_u64(seed as u64 ^ 0x9E37_79B9_7F4A_7C15);

    let mut centroids = if use_kpp {
        kpp_init(xs, n, d, k, &mut rng)
    } else {
        random_init(xs, n, d, k, &mut rng)
    };

    let mut labels = vec![0i32; n];
    let mut sums   = vec![0f32; k * d];
    let mut counts = vec![0u32;  k];

    for _ in 0..max_iter {
        // assignment
        let mut changed = false;
        for i in 0..n {
            let p = &xs[i * d..(i + 1) * d];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let centroid = &centroids[c * d..(c + 1) * d];
                let dd = dist2(p, centroid);
                if dd < best_d {
                    best_d = dd;
                    best = c;
                }
            }
            if labels[i] != best as i32 {
                labels[i] = best as i32;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // update
        sums.iter_mut().for_each(|s| *s = 0.0);
        counts.iter_mut().for_each(|c| *c = 0);
        for i in 0..n {
            let c = labels[i] as usize;
            counts[c] += 1;
            let p = &xs[i * d..(i + 1) * d];
            for j in 0..d {
                sums[c * d + j] += p[j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cf = counts[c] as f32;
                for j in 0..d {
                    centroids[c * d + j] = sums[c * d + j] / cf;
                }
            }
        }
    }

    labels
}

/// Run K-Means and return the full iteration history as a flat f32 buffer.
///
/// Header (first 4 floats):
///   [iter_count (as f32), converged (1.0 or 0.0), k (as f32), d (as f32)]
///
/// Then `iter_count + 1` "snapshots" (initial state + after each iter). Each snapshot:
///   k*d centroid floats, then n label floats (labels stored as f32; JS Math.round).
///
/// Total length: 4 + (iter_count + 1) * (k*d + n)
#[wasm_bindgen]
pub fn kmeans_fit_steps(
    xs: &[f32],
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    seed: u32,
    use_kpp: bool,
) -> Vec<f32> {
    if n == 0 || k == 0 || d == 0 || xs.len() != n * d {
        return vec![];
    }

    let mut rng = Pcg64Mcg::seed_from_u64(seed as u64 ^ 0x9E37_79B9_7F4A_7C15);

    let mut centroids = if use_kpp {
        kpp_init(xs, n, d, k, &mut rng)
    } else {
        random_init(xs, n, d, k, &mut rng)
    };

    let mut labels = vec![0i32; n];
    let mut sums   = vec![0f32; k * d];
    let mut counts = vec![0u32;  k];

    // Helper: push one snapshot (k*d centroids then n labels) into `out`.
    let push_snapshot = |out: &mut Vec<f32>, centroids: &[f32], labels: &[i32]| {
        out.extend_from_slice(centroids);
        for &l in labels {
            out.push(l as f32);
        }
    };

    // Snapshot 0: initial assignment before any update step.
    // Assign labels to the initial centroids first.
    for i in 0..n {
        let p = &xs[i * d..(i + 1) * d];
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..k {
            let centroid = &centroids[c * d..(c + 1) * d];
            let dd = dist2(p, centroid);
            if dd < best_d {
                best_d = dd;
                best = c;
            }
        }
        labels[i] = best as i32;
    }

    // Reserve: header + upper-bound snapshots, avoiding repeated reallocations.
    let snapshot_len = k * d + n;
    let mut out: Vec<f32> = Vec::with_capacity(4 + (max_iter + 1) * snapshot_len);
    // Header placeholder — will be filled in at the end.
    out.extend_from_slice(&[0.0f32; 4]);

    push_snapshot(&mut out, &centroids, &labels);

    let mut iter_count = 0usize;
    let mut converged  = false;

    for _ in 0..max_iter {
        // Update centroids.
        sums.iter_mut().for_each(|s| *s = 0.0);
        counts.iter_mut().for_each(|c| *c = 0);
        for i in 0..n {
            let c = labels[i] as usize;
            counts[c] += 1;
            let p = &xs[i * d..(i + 1) * d];
            for j in 0..d {
                sums[c * d + j] += p[j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let cf = counts[c] as f32;
                for j in 0..d {
                    centroids[c * d + j] = sums[c * d + j] / cf;
                }
            }
        }

        // Reassign labels.
        let mut changed = false;
        for i in 0..n {
            let p = &xs[i * d..(i + 1) * d];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let centroid = &centroids[c * d..(c + 1) * d];
                let dd = dist2(p, centroid);
                if dd < best_d {
                    best_d = dd;
                    best = c;
                }
            }
            if labels[i] != best as i32 {
                labels[i] = best as i32;
                changed = true;
            }
        }

        iter_count += 1;
        push_snapshot(&mut out, &centroids, &labels);

        if !changed {
            converged = true;
            break;
        }
    }

    // Write header.
    out[0] = iter_count as f32;
    out[1] = if converged { 1.0 } else { 0.0 };
    out[2] = k as f32;
    out[3] = d as f32;

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny reproducible dataset: 20 points, 2-D, 3 clusters, seed 42.
    fn fixture() -> (Vec<f32>, usize, usize, usize, usize, u32, bool) {
        let xs: Vec<f32> = (0..20u32)
            .flat_map(|i| [(i % 5) as f32, (i / 5) as f32])
            .collect();
        (xs, 20, 2, 3, 10, 42, true)
    }

    #[test]
    fn test_steps_buffer_layout_header() {
        let (xs, n, d, k, max_iter, seed, use_kpp) = fixture();
        let buf = kmeans_fit_steps(&xs, n, d, k, max_iter, seed, use_kpp);
        // Header must be at least 4 floats.
        assert!(buf.len() >= 4, "buffer too short for header");
        let iter_count = buf[0] as usize;
        let converged  = buf[1];
        let bk         = buf[2] as usize;
        let bd         = buf[3] as usize;
        assert!(iter_count <= max_iter, "iter_count exceeds max_iter");
        assert!(converged == 0.0 || converged == 1.0, "converged must be 0 or 1");
        assert_eq!(bk, k, "k in header mismatch");
        assert_eq!(bd, d, "d in header mismatch");
    }

    #[test]
    fn test_steps_final_state_matches_fit() {
        let (xs, n, d, k, max_iter, seed, use_kpp) = fixture();
        let buf    = kmeans_fit_steps(&xs, n, d, k, max_iter, seed, use_kpp);
        let labels = kmeans_fit(&xs, n, d, k, max_iter, seed, use_kpp);

        let iter_count   = buf[0] as usize;
        let snapshot_len = k * d + n;
        // Last snapshot starts after header + iter_count full snapshots.
        let last_start   = 4 + iter_count * snapshot_len;
        // Labels are the trailing n floats of the last snapshot.
        let label_start  = last_start + k * d;

        assert!(buf.len() >= label_start + n, "buffer too short for final labels");
        for i in 0..n {
            let step_label = buf[label_start + i].round() as i32;
            assert_eq!(step_label, labels[i],
                "label mismatch at point {i}: steps={step_label}, fit={}", labels[i]);
        }
    }

    #[test]
    fn test_steps_centroid_count_per_snapshot() {
        let (xs, n, d, k, max_iter, seed, use_kpp) = fixture();
        let buf = kmeans_fit_steps(&xs, n, d, k, max_iter, seed, use_kpp);

        let iter_count   = buf[0] as usize;
        let num_snapshots = iter_count + 1;
        let snapshot_len  = k * d + n;
        assert_eq!(
            buf.len(),
            4 + num_snapshots * snapshot_len,
            "total buffer length mismatch"
        );
        // Spot-check: centroid values in snapshot 0 must be finite.
        let s0_centroids = &buf[4..4 + k * d];
        for &v in s0_centroids {
            assert!(v.is_finite(), "centroid value is not finite: {v}");
        }
    }

    #[test]
    fn test_steps_label_count_per_snapshot() {
        let (xs, n, d, k, max_iter, seed, use_kpp) = fixture();
        let buf = kmeans_fit_steps(&xs, n, d, k, max_iter, seed, use_kpp);

        let iter_count   = buf[0] as usize;
        let snapshot_len = k * d + n;
        // Check every snapshot's label section.
        for s in 0..=iter_count {
            let label_start = 4 + s * snapshot_len + k * d;
            for i in 0..n {
                let lbl = buf[label_start + i].round() as usize;
                assert!(lbl < k,
                    "label {lbl} out of range [0,{k}) in snapshot {s}, point {i}");
            }
        }
    }

    #[test]
    fn test_steps_converged_flag_when_under_max_iter() {
        // Use large max_iter so the algorithm is likely to converge early.
        let (xs, n, d, k, _, seed, use_kpp) = fixture();
        let buf = kmeans_fit_steps(&xs, n, d, k, 200, seed, use_kpp);

        let iter_count = buf[0] as usize;
        let converged  = buf[1];
        if iter_count < 200 {
            assert_eq!(converged, 1.0,
                "converged should be 1.0 when iter_count ({iter_count}) < max_iter (200)");
        }
    }
}
