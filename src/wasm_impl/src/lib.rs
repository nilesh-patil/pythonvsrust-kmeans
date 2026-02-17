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
