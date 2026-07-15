# Dual-GPU inference for `run_chagos_DGS_2000_to_2025.m`

File-level dual-GPU concurrent inference: a year's remaining files are **routed
by length** into two sets processed **at the same time** by two parallel
workers, each pinned to a distinct GPU (e.g. RTX 4090 + T550). The largest
files go to the high-memory primary GPU (which runs them at full batch size);
the smallest go to the low-memory secondary GPU. Off by default; the single-GPU
serial path is unchanged and is the tested fallback.

## Enabling / knobs (USER INPUT block)

| Knob | Default | Meaning |
|---|---|---|
| `enableDualGpu` | `false` | Turn dual-GPU on. Requires ≥ 2 GPUs; otherwise the serial path runs. |
| `gpuPrimaryFileFraction` | `0.6` | Fraction of a year's **remaining** files given to the primary GPU (`gpuChain(1)`, the largest) — selected **by size**: the primary gets the largest `fraction` of the files, the secondary the smallest rest. |

To turn on: set `enableDualGpu = true`. Tune `gpuPrimaryFileFraction` from the
per-worker file rates in the log so both workers finish together (inference is
Thunderbolt-latency-bound, so the T550 may pull more weight than its raw specs
suggest — measure, don't assume). In a mostly-long year, **raise** the fraction
so the secondary gets fewer, shorter files (it is slow on long files: its 4 GB
forces adaptive batch reduction); in a mostly-short year the two GPUs can share
more evenly.

## How it works

- **Length routing** (`planLengthRoutedSplit`): the remaining files are sorted
  by size (bytes — a proxy for duration, free from `dir()`, no `audioinfo`
  read), and worker A (primary) gets the largest `gpuPrimaryFileFraction` of
  them while worker B (secondary) gets the smallest rest. The split is derived
  only from the **fixed** file sizes and the fraction (ties broken by global
  index), so each worker's file **set** is identical across restarts — which is
  what lets each worker resume its own cache. It is clamped so the secondary
  always gets ≥ 1 (the shortest) file. The two sets are in general
  **non-contiguous** in global index.
- Each worker runs the **shared** per-file pipeline (`runInferenceFileLoop`) —
  the same code as the serial path — over its file list, with **local**
  indexing, into its **own** worker-scoped partial cache
  (`%TEMP%\GAVDNet\detector_raw_partial_<year>_gpuA.mat` / `_gpuB.mat`).
- On completion the client merges (`scatterStructArrays`): preloaded/cached
  results + worker A + worker B, each **scattered to its true global indices**
  (a plain concatenation would not preserve order because the routed sets are
  non-contiguous), into one global `results` array in original file order, then
  the normal serial postprocessing runs.
- **Caveat:** because the routing depends on file sizes, if a file's size
  changes between runs (e.g. a partial copy finishes) the sort — and a worker's
  set — can change, invalidating that worker's cache (it restarts from scratch;
  no data is lost, only recomputed). This is the same class of fragility as the
  file-list dependence and is expected to be rare in a static archive.
- **Resume**: each worker resumes independently from its own cache; the split
  being deterministic is what makes this safe. The existing serial 2006 cache
  is read only as the preload and is migrated to the sharded format on first
  use (see the A4 sharded-cache logic in `loadResultsFromShardedCache`).

## eGPU stability

- Each worker holds a CUDA context on its **own** GPU only; its device
  fallback chain is a single device, so on failure it drops to **CPU**, never
  onto the other worker's GPU (a second context on the Thunderbolt 4090 has
  been observed to destabilise the link).
- The client releases its GPU context (`gpuDevice([])`) before the workers
  pin their devices.
- **If you see eGPU instability**, set `enableDualGpu = false` and use the
  serial path (unaffected). Ideally launch dual-GPU from a fresh MATLAB so the
  client never established a 4090 context before the worker pins it.

## Numerical equivalence (important)

The raw probabilities saved to disk are the durable artefact — they are
re-thresholded in postprocessing sweeps — so dual-GPU must reproduce them, not
merely agree on detections at one activation threshold.

- **Compute-thread count matters.** Parallel workers default to **one**
  compute thread; the serial client uses all cores (`gpuConfig` sets
  `'automatic'`). A mismatch changes the CPU STFT/mel reduction order and
  perturbs raw probabilities by up to ~0.13 at near-zero file-tail bins.
  `runYearDualGpu` therefore sets `maxNumCompThreads('automatic')` in each
  worker to match the serial path.
- **With matched threads** (validated on 24 files of 2006):
  - **Primary-GPU worker (4090) is bit-identical to serial** (max diff `0`).
  - **Secondary-GPU worker (T550)** differs only by the cross-GPU hardware
    floor: median `4.5e-6`, p99 `8.9e-5`, **max `3.7e-4`** — below the postproc
    hysteresis band (AT 0.70 / DT 0.699 = `1e-3`). Zero threshold-crossing
    disagreements at 0.3–0.9; at the extreme 0.1, only isolated single-bin
    flips, which the length threshold (`LT`) rejects. No detection changes at
    any threshold.
- **Acceptance criterion**: `max |serial − dual| < 1e-3` (threshold-independent;
  bounds the worst-case detection change at any future threshold). Median /
  percentiles are reported for characterisation but are *not* the gate — a
  median over tens of thousands of bins hides the tail outliers that a
  re-threshold could act on.

## Validation procedure

1. **Smoke test** (correctness + pinning + raw-prob equivalence):
   ```matlab
   test_dualGpu_smoke        % Utilitiy Scripts/, 24 files, ~few min
   ```
   Confirm: all checks PASS; two `pinned to GPU device …` lines name
   **different** GPUs; no eGPU instability.
2. **Resume test (automated)** — end-to-end, no interruption needed:
   ```matlab
   test_dualGpu_resume       % Utilitiy Scripts/, runs a reference + a resumed pass
   ```
   Runs dual-GPU, seeds each worker's cache to a mid-range state, reruns, and
   checks the resumed result is bit-identical to the uninterrupted reference
   and that each worker resumed from the middle (not from file 1).
3. **Resume test (manual, the real Ctrl-C path)**: `enableDualGpu = true`,
   point `years` at a tiny test year (or a copied folder of ~40 files), start,
   Ctrl-C mid-run, restart → the relaunch prints `resume local <k>` (k > 1) for
   each worker and continues, rather than restarting from file 1.
4. Then run production with `enableDualGpu = true`.

## Files

`Functions/runInferenceFileLoop.m` (shared per-file loop),
`Functions/runYearDualGpu.m` (orchestrator: `parpool(2)` + `spmd`, pinning,
merge), `Functions/planLengthRoutedSplit.m` (length-based file routing),
`Functions/scatterStructArrays.m` (global-index merge of the non-contiguous
worker results), and the externalised helpers `saveResultsToPartialCache.m`,
`loadResultsFromShardedCache.m`, `deletePartialCache.m`, `stepGpuFallback.m`,
`currentFailureThreshold.m`. Tests: `Utilitiy Scripts/test_dualGpu_smoke.m`,
`test_planLengthRoutedSplit.m`, `test_scatterStructArrays.m`.

Ben Jancovich, 2025 — Centre for Marine Science and Innovation, UNSW Sydney.
