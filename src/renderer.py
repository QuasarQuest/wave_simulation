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
MAX_MESH_SUBDIV = 256  # cap render mesh density regardless of sim grid size


class Renderer:
    def __init__(self, rp: RenderParams, grid_n: int):
        self.rp = rp
        rl.set_config_flags(rl.FLAG_MSAA_4X_HINT)
        rl.init_window(rp.width, rp.height, b"FFT Ocean -- NumPy vs CuPy")
        rl.set_target_fps(rp.target_fps)

        # Ocean shader + uniform locations.
        vs = os.path.join(SHADER_DIR, "ocean.vs")
        fs = os.path.join(SHADER_DIR, "ocean.fs")
        self.shader = rl.load_shader(vs, fs)
        self._loc = {
            name: rl.get_shader_location(self.shader, name)
            for name in ("uVScale", "uHScale", "uSunDir", "uCamPos",
                         "uDeep", "uShallow", "uSky", "uSunCol",
                         "uGridN", "uWorldSize")
        }
        self._set_static_uniforms()

        # How many patch copies to tile around the centre (radius 1 -> 3x3).
        self.tile_radius = 1

        # Orbit camera state (spherical around the patch centre).
        self.yaw = math.radians(35.0)
        self.pitch = math.radians(24.0)
        self.distance = rp.world_size * 1.5
        self.target = rl.Vector3(0.0, 0.0, 0.0)
        self.camera = rl.Camera3D(
            rl.Vector3(0, 50, 100), self.target, rl.Vector3(0, 1, 0),
            45.0, rl.CAMERA_PERSPECTIVE,
        )

        self.model = None
        self.tex = None
        self._tex_n = 0
        self.rebuild(grid_n)

    # ----------------------------------------------------------- gpu objects
    def rebuild(self, grid_n: int) -> None:
        """(Re)build the mesh + displacement texture for a new grid size."""
        if self.model is not None:
            rl.unload_model(self.model)
        if self.tex is not None:
            rl.unload_texture(self.tex)

        subdiv = min(grid_n, MAX_MESH_SUBDIV)
        mesh = rl.gen_mesh_plane(self.rp.world_size, self.rp.world_size,
                                 subdiv, subdiv)
        self.model = rl.load_model_from_mesh(mesh)
        self.model.materials[0].shader = self.shader

        # Float RGBA displacement texture, created from a zero image then
        # updated every frame via update_texture.
        self._tex_n = grid_n
        self._zero = np.zeros((grid_n, grid_n, 4), dtype=np.float32)
        img = rl.Image()
        img.data = ffi.cast("void *", self._zero.ctypes.data)
        img.width = grid_n
        img.height = grid_n
        img.mipmaps = 1
        img.format = rl.PIXELFORMAT_UNCOMPRESSED_R32G32B32A32
        self.tex = rl.load_texture_from_image(img)
        rl.set_texture_filter(self.tex, rl.TEXTURE_FILTER_BILINEAR)
        # REPEAT so neighbour-texel normal sampling wraps seamlessly across
        # tiled patches (the height field is periodic).
        rl.set_texture_wrap(self.tex, rl.TEXTURE_WRAP_REPEAT)
        # Sample normals at the *mesh* resolution (not the finer texture res) so
        # the sun glint doesn't moire/alias against the coarser displaced
        # geometry. dWorld in the shader uses the same spacing for consistency.
        self._float("uGridN", float(subdiv))
        self._float("uWorldSize", self.rp.world_size)
        # Bind it to the material's albedo slot -> shader sampler "texture0".
        self.model.materials[0].maps[rl.MATERIAL_MAP_ALBEDO].texture = self.tex

    def upload(self, packed: np.ndarray) -> None:
        """Push the packed (N,N,4) float32 fields to the GL texture."""
        if packed.shape[0] != self._tex_n:
            return
        c = np.ascontiguousarray(packed, dtype=np.float32)
        rl.update_texture(self.tex, ffi.cast("void *", c.ctypes.data))

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

    def set_height_scale(self, v: float) -> None:
        self.rp.height_scale = v
        self._float("uVScale", v)
        self._float("uHScale", v * 0.22)

    # ----------------------------------------------------------------- camera
    def update_camera(self, allow_mouse: bool) -> None:
        if allow_mouse:
            if rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT):
                md = rl.get_mouse_delta()
                self.yaw -= md.x * 0.005
                self.pitch = max(0.05, min(1.5, self.pitch + md.y * 0.005))
            self.distance *= (1.0 - rl.get_mouse_wheel_move() * 0.08)
            self.distance = max(self.rp.world_size * 0.25,
                                min(self.rp.world_size * 3.0, self.distance))
        d, p, y = self.distance, self.pitch, self.yaw
        self.camera.position = rl.Vector3(
            d * math.cos(p) * math.cos(y),
            d * math.sin(p),
            d * math.cos(p) * math.sin(y),
        )

    # ------------------------------------------------------------------ frame
    def begin_world(self) -> None:
        self._vec3("uCamPos", self.camera.position.x,
                   self.camera.position.y, self.camera.position.z)
        rl.begin_drawing()
        rl.clear_background(rl.Color(150, 178, 235, 255))
        rl.begin_mode_3d(self.camera)
        # The height field is periodic, so tiling the patch is seamless and
        # makes it read as an open ocean rather than a lone floating square.
        s = self.rp.world_size
        r = self.tile_radius
        for iz in range(-r, r + 1):
            for ix in range(-r, r + 1):
                rl.draw_model(self.model, rl.Vector3(ix * s, 0, iz * s),
                              1.0, rl.WHITE)
        rl.end_mode_3d()

    def end_frame(self) -> None:
        rl.end_drawing()

    def should_close(self) -> bool:
        return rl.window_should_close()

    def close(self) -> None:
        if self.model is not None:
            rl.unload_model(self.model)
        if self.tex is not None:
            rl.unload_texture(self.tex)
        rl.unload_shader(self.shader)
        rl.close_window()
