"""Headless CPU-vs-GPU benchmark of the ocean step across grid sizes."""

from __future__ import annotations

import numpy as np

from config import OceanParams, GRID_SIZES
from src.backend import make_backend, cupy_available
from src.ocean import Ocean


def _time_backend(kind: str, params: OceanParams, steps: int = 40):
    backend, _ = make_backend(kind)
    ocean = Ocean(backend, params)
    ocean.step(0.0)  # warmup (JIT / cuFFT plan)
    with backend.time_block() as tm:
        for i in range(steps):
            ocean.step(i * 0.05)
    return tm.elapsed / steps * 1000.0  # ms/step


def run_benchmark(base: OceanParams) -> None:
    gpu_ok, msg = cupy_available()
    print(f"GPU: {'available -- ' + msg if gpu_ok else 'NOT available'}\n")
    print(f"{'grid':>6} | {'numpy ms':>9} | {'cupy ms':>9} | {'speedup':>8} | match")
    print("-" * 56)
    for n in GRID_SIZES:
        p = base.copy()
        p.N = n
        cpu = _time_backend("numpy", p)
        if gpu_ok:
            gpu = _time_backend("cupy", p)
            # correctness: identical surface CPU vs GPU
            bc, _ = make_backend("numpy"); oc = Ocean(bc, p); oc.step(3.0)
            bg, _ = make_backend("cupy"); og = Ocean(bg, p); og.step(3.0)
            diff = float(np.abs(bc.to_cpu(oc.height) - bg.to_cpu(og.height)).max())
            print(f"{n:>6} | {cpu:>9.2f} | {gpu:>9.2f} | {cpu/gpu:>7.1f}x | "
                  f"{'ok' if diff < 1e-3 else f'DIFF {diff:.2e}'}")
        else:
            print(f"{n:>6} | {cpu:>9.2f} | {'--':>9} | {'--':>8} |  --")


if __name__ == "__main__":
    run_benchmark(OceanParams())
