
"""FFT Ocean simulation -- NumPy (CPU) vs CuPy (GPU) benchmark with a 3D GUI.

Implements the height-field ocean from doc/coursenotes2004.pdf (Tessendorf,
"Simulating Ocean Water"): a Phillips-spectrum field advanced by the deep-water
dispersion relation and inverse-FFT'd every frame. The same backend-agnostic
code runs on NumPy or CuPy so you can watch the GPU speedup live.

Run:  .venv/bin/python main.py [--backend numpy|cupy] [--grid N] [--headless]
"""

from __future__ import annotations

import argparse
import os

from config import AppConfig, GRID_SIZES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["numpy", "cupy"], default=None,
                    help="start on this backend (default: cupy, falls back to numpy)")
    ap.add_argument("--grid", type=int, choices=GRID_SIZES, default=None,
                    help="initial grid size NxN")
    ap.add_argument("--headless", action="store_true",
                    help="run the CPU-vs-GPU benchmark without opening a window")
    ap.add_argument("--nvidia-gl", action="store_true",
                    help="force the OpenGL context onto the NVIDIA GPU via PRIME "
                         "render-offload (hybrid laptops). Same GPU as CuPy, but "
                         "compute and rendering then contend for one GPU.")
    args = ap.parse_args()

    if args.nvidia_gl:
        # Must be set before any GL init (i.e. before importing the renderer).
        os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

    cfg = AppConfig()
    if args.backend:
        cfg.backend = args.backend
    if args.grid:
        cfg.ocean.N = args.grid

    if args.headless:
        from src.benchmark import run_benchmark
        run_benchmark(cfg.ocean)
        return

    from src.app import App
    App(cfg).run()


if __name__ == "__main__":
    main()
