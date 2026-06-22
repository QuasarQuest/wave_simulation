"""raylib 3D renderer for the ocean surface.

A flat plane mesh is displaced in the vertex shader by sampling a per-frame
"displacement texture" packed by the Ocean (r=height, g=Dx, b=Dz). Normals are
derived per-fragment from screen-space derivatives, so the only per-frame upload
is that one texture -- keeping the bottleneck on the simulation, which is the
whole point of the CPU-vs-GPU comparison.

The render mesh density is capped independently of the simulation resolution so
that bumping the grid to 1024/2048 stresses the *FFT*, not the rasteriser.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pyray as rl
from raylib import ffi

from config import RenderParams

SHADER_DIR = os.path.join(os.path.dirname(__file__), "..", "shaders")
# Render mesh density per patch (capped, independent of the sim grid size) and
# how many patch copies span the visible ocean (TILES x TILES).
SUBDIV_PER_PATCH = 150
TILES = 3
# raylib's gen_mesh_plane uses 16-bit indices, so a plane with more than 65536
# vertices ((subdiv+1)^2) overflows and only part of it renders -- that's the
# "cut off" ocean. 255 -> exactly 256*256 = 65536 verts, the most one plane can
# hold, so the whole finite patch draws.
MAX_SUBDIV = 255


class Renderer:
    def __init__(self, rp: RenderParams, grid_n: int):
        self.rp = rp
        # Prefer X11/XWayland over native Wayland. raylib 6.1-dev's Wayland
        # HiDPI path renders into a partial viewport (3/4 of the window black)
        # and mis-maps the mouse; under X11 render==screen with a sane scale and
        # everything lines up. This hint must precede init_window (it configures
        # the glfwInit that InitWindow performs). Set OCEAN_FORCE_WAYLAND=1 to
        # keep native Wayland.
        if os.environ.get("OCEAN_FORCE_WAYLAND") != "1":
            try:
                rl.glfw_init_hint(rl.GLFW_PLATFORM, rl.GLFW_PLATFORM_X11)
            except Exception:
                pass  # non-GLFW/headless builds: ignore
        # NOTE: FLAG_WINDOW_HIGHDPI is deliberately NOT set. On this raylib
        # (6.1-dev) + Wayland build it produces a 2x framebuffer but only draws
        # into a logical-sized viewport at the corner (3/4 of the window stays
        # black) and the fix (forcing rlViewport) doesn't stick. Without it,
        # raylib renders at logical resolution (the compositor upscales) and
        # drawing + mouse share one coordinate space, which actually works.
        rl.set_config_flags(rl.FLAG_MSAA_4X_HINT | rl.FLAG_WINDOW_RESIZABLE)
        rl.init_window(rp.width, rp.height, b"FFT Ocean -- NumPy vs CuPy")

        # The window may open tiny relative to a HiDPI monitor. maximize_window
        # is honoured by Wayland (unlike set_window_position / reliable resize).
        # The GUI is scaled per-frame from the *actual* framebuffer size (see
        # App._draw_gui), keeping widgets and the mouse in one coordinate space
        # regardless of how big the compositor makes the window.
        mon = rl.get_current_monitor()
        mw, mh = rl.get_monitor_width(mon), rl.get_monitor_height(mon)
        rl.set_window_size(int(mw * 0.85), int(mh * 0.85))
        rl.maximize_window()
        # Pump a few frames so the compositor settles the resize, then report
        # the coordinate spaces (helps diagnose HiDPI/Wayland mismatches).
        for _ in range(8):
            rl.begin_drawing()
            rl.end_drawing()
        sc = rl.get_window_scale_dpi()
        print(f"[DIAG] monitor={mw}x{mh}  screen={rl.get_screen_width()}x"
              f"{rl.get_screen_height()}  render={rl.get_render_width()}x"
              f"{rl.get_render_height()}  scale_dpi={sc.x:.2f}x{sc.y:.2f}",
              flush=True)
        # Push the far clip plane out so a zoomed-out ocean is never clipped
        # (raylib's default far plane is only 1000).
        rl.rl_set_clip_planes(0.05, 12000.0)
        rl.set_target_fps(rp.target_fps)

        # Ocean shader + uniform locations.
        vs = os.path.join(SHADER_DIR, "ocean.vs")
        fs = os.path.join(SHADER_DIR, "ocean.fs")
        self.shader = rl.load_shader(vs, fs)
        self._loc = {
            name: rl.get_shader_location(self.shader, name)
            for name in ("uVScale", "uHScale", "uSlopeScale", "uUVRepeat",
                         "uSunDir", "uCamPos", "uDeep", "uShallow", "uSky",
                         "uSunCol")
        }
        self._set_static_uniforms()

        # Orbit camera state (spherical around the patch centre). Framed so the
        # whole TILES x TILES ocean fits in view (span = world_size * TILES).
        self.span = rp.world_size * TILES
        self.yaw = math.radians(30.0)
        self.pitch = math.radians(33.0)
        self.distance = self.span * 1.1
        self.target = rl.Vector3(0.0, 0.0, 0.0)
        self.camera = rl.Camera3D(
            rl.Vector3(0, 50, 100), self.target, rl.Vector3(0, 1, 0),
            45.0, rl.CAMERA_PERSPECTIVE,
        )

        self.model = None
        self.tex = None       # displacement: (height, Dx, Dz)
        self.tex_n = None     # slopes: (dH/dx, dH/dz)
        self._tex_n_res = 0
        self.rebuild(grid_n)

    # ----------------------------------------------------------- gpu objects
    @staticmethod
    def _make_float_texture(grid_n: int):
        """Create a zero-initialised RGBA32F texture, bilinear + REPEAT wrap."""
        zero = np.zeros((grid_n, grid_n, 4), dtype=np.float32)
        img = rl.Image()
        img.data = ffi.cast("void *", zero.ctypes.data)
        img.width = grid_n
        img.height = grid_n
        img.mipmaps = 1
        img.format = rl.PIXELFORMAT_UNCOMPRESSED_R32G32B32A32
        tex = rl.load_texture_from_image(img)
        rl.set_texture_filter(tex, rl.TEXTURE_FILTER_BILINEAR)
        rl.set_texture_wrap(tex, rl.TEXTURE_WRAP_REPEAT)  # seamless tiling
        return tex

    def rebuild(self, grid_n: int) -> None:
        """(Re)build the mesh + displacement/slope textures for a new grid."""
        if self.model is not None:
            rl.unload_model(self.model)
        if self.tex is not None:
            rl.unload_texture(self.tex)
        if self.tex_n is not None:
            rl.unload_texture(self.tex_n)

        # One big continuous plane spanning TILES x TILES patches. UVs are
        # scaled by TILES in the vertex shader so the texture repeats with no
        # internal mesh edges (hence no seams/gaps).
        span = self.rp.world_size * TILES
        subdiv = min(min(grid_n, SUBDIV_PER_PATCH) * TILES, MAX_SUBDIV)
        mesh = rl.gen_mesh_plane(span, span, subdiv, subdiv)
        self.model = rl.load_model_from_mesh(mesh)
        self.model.materials[0].shader = self.shader
        self._float("uUVRepeat", float(TILES))

        self._tex_n_res = grid_n
        self.tex = self._make_float_texture(grid_n)     # displacement
        self.tex_n = self._make_float_texture(grid_n)   # slopes
        # texture0 = albedo slot (displacement), texture2 = normal slot (slopes).
        self.model.materials[0].maps[rl.MATERIAL_MAP_ALBEDO].texture = self.tex
        self.model.materials[0].maps[rl.MATERIAL_MAP_NORMAL].texture = self.tex_n

    def upload(self, packed: np.ndarray, packed_n: np.ndarray) -> None:
        """Push the displacement and slope fields to their GL textures."""
        if packed.shape[0] != self._tex_n_res:
            return
        c = np.ascontiguousarray(packed, dtype=np.float32)
        cn = np.ascontiguousarray(packed_n, dtype=np.float32)
        rl.update_texture(self.tex, ffi.cast("void *", c.ctypes.data))
        rl.update_texture(self.tex_n, ffi.cast("void *", cn.ctypes.data))

    # --------------------------------------------------------------- uniforms
    def _vec3(self, loc_name: str, x, y, z) -> None:
        rl.set_shader_value(self.shader, self._loc[loc_name],
                            ffi.new("float[]", [float(x), float(y), float(z)]),
                            rl.SHADER_UNIFORM_VEC3)

    def _float(self, loc_name: str, v) -> None:
        rl.set_shader_value(self.shader, self._loc[loc_name],
                            ffi.new("float[]", [float(v)]),
                            rl.SHADER_UNIFORM_FLOAT)

    def _set_static_uniforms(self) -> None:
        rp = self.rp
        self._vec3("uSunDir", *rp.sun_dir)
        self._vec3("uDeep", *rp.deep_color)
        self._vec3("uShallow", *rp.shallow_color)
        self._vec3("uSky", *rp.sky_color)
        self._vec3("uSunCol", *rp.sun_color)
        self._float("uVScale", rp.height_scale)
        self._float("uHScale", rp.height_scale * 0.22)
        self._float("uSlopeScale", rp.height_scale * 1.2)

    def set_height_scale(self, v: float) -> None:
        self.rp.height_scale = v
        self._float("uVScale", v)
        self._float("uHScale", v * 0.22)
        self._float("uSlopeScale", v * 1.2)

    # ----------------------------------------------------------------- camera
    def update_camera(self, allow_mouse: bool) -> None:
        if allow_mouse:
            if rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT):
                md = rl.get_mouse_delta()
                self.yaw -= md.x * 0.005
                self.pitch = max(0.05, min(1.5, self.pitch + md.y * 0.005))
            # RMB drag pans the look-at point across the water, in the camera's
            # horizontal frame; speed scales with zoom so it feels consistent.
            if rl.is_mouse_button_down(rl.MOUSE_BUTTON_RIGHT):
                md = rl.get_mouse_delta()
                speed = self.distance * 0.0015
                cy, sy = math.cos(self.yaw), math.sin(self.yaw)
                self.target.x -= (sy * md.x + cy * md.y) * speed
                self.target.z -= (-cy * md.x + sy * md.y) * speed
            self.distance *= (1.0 - rl.get_mouse_wheel_move() * 0.08)
            self.distance = max(self.span * 0.2,
                                min(self.span * 1.8, self.distance))
        d, p, y = self.distance, self.pitch, self.yaw
        # Orbit + zoom around the (movable) look-at point.
        self.camera.position = rl.Vector3(
            self.target.x + d * math.cos(p) * math.cos(y),
            self.target.y + d * math.sin(p),
            self.target.z + d * math.cos(p) * math.sin(y),
        )
        self.camera.target = self.target

    # ------------------------------------------------------------------ frame
    def begin_world(self) -> None:
        self._vec3("uCamPos", self.camera.position.x,
                   self.camera.position.y, self.camera.position.z)
        rl.begin_drawing()
        rl.clear_background(rl.Color(150, 178, 235, 255))
        rl.begin_mode_3d(self.camera)
        rl.draw_model(self.model, rl.Vector3(0, 0, 0), 1.0, rl.WHITE)
        rl.end_mode_3d()

    @staticmethod
    def end_frame() -> None:
        rl.end_drawing()

    @staticmethod
    def should_close() -> bool:
        return rl.window_should_close()

    def close(self) -> None:
        if self.model is not None:
            rl.unload_model(self.model)
        if self.tex is not None:
            rl.unload_texture(self.tex)
        if self.tex_n is not None:
            rl.unload_texture(self.tex_n)
        rl.unload_shader(self.shader)
        rl.close_window()
