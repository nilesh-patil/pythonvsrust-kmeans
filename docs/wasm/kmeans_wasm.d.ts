/* tslint:disable */
/* eslint-disable */

export function init_panic_hook(): void;

export function kmeans_fit(xs: Float32Array, n: number, d: number, k: number, max_iter: number, seed: number, use_kpp: boolean): Int32Array;

/**
 * Run K-Means and return the full iteration history as a flat f32 buffer.
 *
 * Header (first 4 floats):
 *   [iter_count (as f32), converged (1.0 or 0.0), k (as f32), d (as f32)]
 *
 * Then `iter_count + 1` "snapshots" (initial state + after each iter). Each snapshot:
 *   k*d centroid floats, then n label floats (labels stored as f32; JS Math.round).
 *
 * Total length: 4 + (iter_count + 1) * (k*d + n)
 */
export function kmeans_fit_steps(xs: Float32Array, n: number, d: number, k: number, max_iter: number, seed: number, use_kpp: boolean): Float32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly init_panic_hook: () => void;
    readonly kmeans_fit: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly kmeans_fit_steps: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
