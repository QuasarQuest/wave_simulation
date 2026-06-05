"""Array-backend abstraction so the *same* ocean code runs on CPU or GPU.

The simulation never imports numpy or cupy directly -- it asks for a `Backend`
and uses `backend.xp` (the array module) for all maths. Swapping CPU<->GPU is
then just constructing a different Backend, which is what the GUI's backend
toggle does.
"""

from __future__ import annotations

import time

import numpy as np


def cupy_available() -> tuple[bool, str]:
    """Return (ok, message). Tries a real allocation + kernel so we catch
    missing-driver / missing-toolkit / no-GPU situations up front."""
    try:
        import cupy as cp  # noqa: F401
    except Exception as e:  # pragma: no cover - import-time failure
        return False, f"cupy not importable: {e}"
    try:
        a = cp.arange(4, dtype=cp.float32)   # forces NVRTC kernel compile
        _ = cp.fft.fft(a)                    # forces cuFFT load
        cp.cuda.Stream.null.synchronize()
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode()
        return True, f"CuPy on {name}"
    except Exception as e:
        return False, f"cupy present but GPU unusable: {e}"


class Backend:
    """Holds the active array module and device-aware helpers."""

    def __init__(self, kind: str):
        if kind == "cupy":
            import cupy as cp

            self.xp = cp
            self.kind = "cupy"
            self.is_gpu = True
            props = cp.cuda.runtime.getDeviceProperties(0)
            self.device_name = props["name"].decode()
        elif kind == "numpy":
            self.xp = np
            self.kind = "numpy"
            self.is_gpu = False
            self.device_name = "CPU (numpy)"
        else:
            raise ValueError(f"unknown backend kind: {kind!r}")

    # -- device <-> host helpers -------------------------------------------

    def to_cpu(self, arr) -> np.ndarray:
        """Return a host numpy array regardless of backend."""
        if self.is_gpu:
            return self.xp.asnumpy(arr)
        return np.asarray(arr)

    def sync(self) -> None:
        """Block until queued GPU work is done (no-op on CPU).

        Essential for honest timing: GPU calls are asynchronous, so without a
        sync we'd measure launch time, not compute time."""
        if self.is_gpu:
            self.xp.cuda.Stream.null.synchronize()

    def time_block(self):
        """Context manager returning elapsed wall-time (seconds) of GPU+CPU
        work inside it, with a sync at the end so GPU timing is accurate."""
        return _Timer(self)


class _Timer:
    def __init__(self, backend: Backend):
        self._b = backend
        self.elapsed = 0.0

    def __enter__(self):
        self._b.sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._b.sync()
        self.elapsed = time.perf_counter() - self._t0
        return False


def make_backend(kind: str) -> tuple[Backend, str]:
    """Construct the requested backend, falling back to numpy if GPU is
    unavailable. Returns (backend, status_message)."""
    if kind == "cupy":
        ok, msg = cupy_available()
        if not ok:
            return Backend("numpy"), f"GPU unavailable, using CPU. ({msg})"
        return Backend("cupy"), msg
    return Backend("numpy"), "CPU (numpy)"
