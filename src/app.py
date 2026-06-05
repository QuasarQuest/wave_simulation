"""Application: main loop, raygui control panel, and live CPU-vs-GPU timing.

Holds the active Backend + Ocean + Renderer and rebuilds them in response to
GUI changes (grid size, backend toggle, wind/amplitude...). Changes requested
inside the GUI are queued and applied at the top of the next frame so we never
create/destroy GL resources mid-draw.
"""

from __future__ import annotations

import pyray as rl
from raylib import ffi

from config import AppConfig, GRID_SIZES, OceanParams
from src.backend import make_backend, cupy_available
from src.ocean import Ocean
from src.renderer import Renderer

PANEL_W = 300


class App:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.params: OceanParams = cfg.ocean

        self.gpu_ok, self.gpu_msg = cupy_available()
        want = cfg.backend if (cfg.backend == "numpy" or self.gpu_ok) else "numpy"
        self.backend, self.backend_msg = make_backend(want)

        self.ocean = Ocean(self.backend, self.params)
        self.renderer = Renderer(cfg.render, self.params.N)

        self.sim_time = 0.0
        self.sim_ms = {"numpy": None, "cupy": None}  # EMA of step() time
        self.last_speedup = None
        self._pending_backend = None
        self._pending_rebuild = False

        # GUI widget state (cffi pointers reused each frame).
        self._grid_idx = GRID_SIZES.index(self.params.N)
        self._backend_idx = 0 if self.backend.kind == "numpy" else 1

    # --------------------------------------------------------- backend / grid
    def _switch_backend(self, kind: str) -> None:
        if kind == self.backend.kind:
            return
        if kind == "cupy" and not self.gpu_ok:
            return
        self.backend, self.backend_msg = make_backend(kind)
        self.ocean = Ocean(self.backend, self.params)
        self._backend_idx = 0 if self.backend.kind == "numpy" else 1

    def _rebuild_grid(self, n: int) -> None:
        self.params.N = n
        self.ocean.update_params(self.params)
        self.renderer.rebuild(n)
        # Timings are per-grid-size; clear so the speedup only compares
        # measurements taken at the same resolution.
        self.sim_ms = {"numpy": None, "cupy": None}
        self.last_speedup = None

    # --------------------------------------------------------------- one step
    def _simulate(self) -> None:
        self.sim_time += rl.get_frame_time() * self.params.time_scale
        with self.backend.time_block() as tm:
            self.ocean.step(self.sim_time)
        ms = tm.elapsed * 1000.0
        prev = self.sim_ms[self.backend.kind]
        self.sim_ms[self.backend.kind] = ms if prev is None else 0.9 * prev + 0.1 * ms
        if self.sim_ms["numpy"] and self.sim_ms["cupy"]:
            self.last_speedup = self.sim_ms["numpy"] / self.sim_ms["cupy"]

        packed = self.ocean.pack_texture()   # GPU->CPU readback happens here
        self.renderer.upload(packed)

    # ------------------------------------------------------------------- gui
    def _draw_gui(self) -> None:
        rl.gui_panel(rl.Rectangle(0, 0, PANEL_W, self.cfg.render.height),
                     b"FFT Ocean Controls")
        x, w = 12, PANEL_W - 24
        y = 36

        def slider(label, value, lo, hi, fmt="%.1f"):
            nonlocal y
            rl.gui_label(rl.Rectangle(x, y, w, 16),
                         f"{label}: {fmt % value}".encode())
            y += 18
            ptr = ffi.new("float *", float(value))
            rl.gui_slider_bar(rl.Rectangle(x, y, w, 16), b"", b"", ptr, lo, hi)
            y += 26
            return ptr[0]

        p = self.params
        before = (p.wind_speed, p.wind_dir_deg, p.amplitude)
        p.wind_speed = slider("Wind speed (m/s)", p.wind_speed, 5.0, 60.0)
        p.wind_dir_deg = slider("Wind dir (deg)", p.wind_dir_deg, 0.0, 360.0)
        p.amplitude = slider("Amplitude", p.amplitude, 0.1, 12.0, "%.2f")
        p.choppiness = slider("Choppiness", p.choppiness, 0.0, 3.0, "%.2f")
        p.time_scale = slider("Time scale", p.time_scale, 0.0, 3.0, "%.2f")

        hs = slider("Height scale", self.cfg.render.height_scale, 50.0, 1000.0, "%.0f")
        if hs != self.cfg.render.height_scale:
            self.renderer.set_height_scale(hs)

        # Apply spectrum-affecting changes (rebuilds inside update_params).
        if before != (p.wind_speed, p.wind_dir_deg, p.amplitude):
            self.ocean.update_params(p)

        # Grid size selector.
        y += 6
        rl.gui_label(rl.Rectangle(x, y, w, 16), b"Grid size (NxN)")
        y += 18
        gi = ffi.new("int *", self._grid_idx)
        rl.gui_toggle_group(rl.Rectangle(x, y, (w - 16) / 5, 22),
                            b";".join(str(g).encode() for g in GRID_SIZES), gi)
        if gi[0] != self._grid_idx:
            self._grid_idx = gi[0]
            self._pending_rebuild = True
        y += 34

        # Backend toggle.
        rl.gui_label(rl.Rectangle(x, y, w, 16), b"Backend")
        y += 18
        bi = ffi.new("int *", self._backend_idx)
        rl.gui_toggle_group(rl.Rectangle(x, y, (w - 4) / 2, 24),
                            b"NumPy (CPU);CuPy (GPU)", bi)
        if bi[0] != self._backend_idx:
            if bi[0] == 1 and not self.gpu_ok:
                pass  # GPU unavailable; ignore
            else:
                self._pending_backend = "numpy" if bi[0] == 0 else "cupy"
        y += 34
        if not self.gpu_ok:
            rl.gui_label(rl.Rectangle(x, y, w, 16), b"(GPU unavailable)")
            y += 20

        # --- readouts ---
        y += 8
        self._readouts(x, w, y)

    def _readouts(self, x, w, y) -> None:
        fps = rl.get_fps()
        sm = self.sim_ms[self.backend.kind]
        lines = [
            ("Device", self.backend.device_name),
            ("FPS", f"{fps}"),
            ("Sim step", f"{sm:.2f} ms" if sm else "--"),
            ("  numpy", f"{self.sim_ms['numpy']:.2f} ms" if self.sim_ms['numpy'] else "--"),
            ("  cupy", f"{self.sim_ms['cupy']:.2f} ms" if self.sim_ms['cupy'] else "--"),
            ("GPU speedup", f"{self.last_speedup:.1f}x" if self.last_speedup else "run both"),
        ]
        for label, val in lines:
            rl.gui_label(rl.Rectangle(x, y, w, 18), f"{label}: {val}".encode())
            y += 20
        rl.gui_label(rl.Rectangle(x, y + 6, w, 18),
                     b"LMB drag: orbit   wheel: zoom")

    # ------------------------------------------------------------------- loop
    def run(self, max_frames: int | None = None, screenshot: str | None = None) -> None:
        frame = 0
        while not self.renderer.should_close():
            if max_frames is not None and frame >= max_frames:
                break
            frame += 1
            # Apply queued GL/backend changes before any drawing this frame.
            if self._pending_backend:
                self._switch_backend(self._pending_backend)
                self._pending_backend = None
            if self._pending_rebuild:
                self._rebuild_grid(GRID_SIZES[self._grid_idx])
                self._pending_rebuild = False

            self._simulate()

            mouse_over_panel = rl.get_mouse_x() < PANEL_W
            self.renderer.update_camera(allow_mouse=not mouse_over_panel)

            self.renderer.begin_world()
            self._draw_gui()
            self.renderer.end_frame()

            if screenshot and frame == max_frames:
                rl.take_screenshot(screenshot.encode())

        self.renderer.close()
