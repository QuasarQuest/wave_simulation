# FFT Ocean Simulation — NumPy (CPU) vs CuPy (GPU)

A real-time 3D ocean-wave simulation implementing the FFT height-field method
from Jerry Tessendorf's **"Simulating Ocean Water"** (`doc/coursenotes2004.pdf`):
a Phillips-spectrum wave field advanced by the deep-water dispersion relation and
inverse-FFT'd every frame.

The same backend-agnostic simulation code runs on **NumPy (CPU)** or **CuPy (GPU)**,
so you can flip between them live and watch the speedup. Rendering is done with
**raylib** (OpenGL) — a single continuous, seamlessly-tiled water surface with a
custom ocean shader (vertex displacement, Fresnel sky reflection, sun glint,
whitecaps) and an orbit camera, plus a `raygui` control panel.

## Highlights

- **Tessendorf FFT ocean**: Phillips spectrum → `h(k,t)` → inverse FFT for height,
  horizontal choppiness displacement, and surface slopes (for exact normals).
- **CPU vs GPU, same code**: a thin `xp` backend (`numpy`/`cupy`) — identical maths,
  one toggle. CPU and GPU produce a bit-for-bit matching surface.
- **Live benchmark**: FPS, per-step sim time for each backend, and the GPU speedup.
- **Configurable grid**: 128 / 256 / 512 / 1024 / 2048, switchable at runtime.
- **60 FPS** at the default 512² on a GTX 1650 Ti.

## Measured speedup (GTX 1650 Ti, full per-frame field incl. slopes)

| grid  | NumPy (CPU) | CuPy (GPU) | speedup |
|------:|------------:|-----------:|--------:|
|  128  |    ~1.0 ms  |   ~0.7 ms  |   1.3×  |
|  256  |    ~3.6 ms  |   ~0.7 ms  |   5.2×  |
|  512  |   ~15.2 ms  |   ~1.2 ms  |  13.2×  |
| 1024  |   ~83.9 ms  |   ~4.3 ms  |  19.8×  |
| 2048  |  ~595.8 ms  |  ~17.0 ms  |  35.1×  |

The GPU advantage grows with grid size (cuFFT scales far better than NumPy's
pocketfft). At 1024²+, the on-screen frame rate is limited not by the simulation
but by the **GPU→CPU→GPU texture transfer** (there is no CUDA–OpenGL interop yet),
which is itself a nice illustration of where the real cost moves.

## Setup

This project was set up on a bare **Python 3.14** with native `cp314` wheels for
everything (numpy, raylib, cupy 14.x). The system Python had no `pip`/`venv`, so
pip was bootstrapped into a venv:

```bash
python3.14 -m venv .venv --without-pip
curl -LsSf https://bootstrap.pypa.io/get-pip.py | .venv/bin/python -
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` installs `cupy-cuda12x[ctk]`; the `[ctk]` extra pulls the CUDA
toolkit headers as wheels (needed for CuPy's NVRTC kernel JIT) — **no system CUDA
toolkit and no sudo required**, just an NVIDIA driver. CuPy is optional: if it (or
a GPU) is unavailable, the app runs CPU-only and disables the GPU toggle.

## Run

```bash
.venv/bin/python main.py                 # GUI, starts on GPU (falls back to CPU)
.venv/bin/python main.py --backend numpy # start on CPU
.venv/bin/python main.py --grid 1024     # start at 1024x1024
.venv/bin/python main.py --headless      # print the CPU-vs-GPU benchmark table
```

**Controls:** left-mouse drag = orbit, wheel = zoom. The panel adjusts wind
speed/direction, amplitude, choppiness, time scale, vertical exaggeration, grid
size, and the active backend.

## Layout

```
config.py          parameters & defaults (grid sizes, wind, colours, camera)
main.py            entry point / CLI
src/backend.py     numpy|cupy abstraction, device<->host helpers, GPU-accurate timing
src/ocean.py       Tessendorf FFT ocean core (backend-agnostic)
src/renderer.py    raylib: one continuous displaced mesh + ocean shader + camera
src/app.py         main loop, raygui panel, live timing/benchmark
src/benchmark.py   headless CPU-vs-GPU benchmark
shaders/ocean.vs   vertex displacement from the height texture
shaders/ocean.fs   normals from FFT slopes, lighting, Fresnel, foam
```
