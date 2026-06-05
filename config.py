"""Default simulation / rendering parameters.

These are the *initial* values; most are tunable live from the GUI panel.
The Tessendorf ocean model is described in doc/coursenotes2004.pdf
("Simulating Ocean Water", Jerry Tessendorf).
"""

from dataclasses import dataclass, field


# Grid sizes selectable from the GUI. Must be powers of two for the FFT.
# 2048 on a 4 GB GTX 1650 Ti is fine for the spectrum, but the CPU (numpy)
# backend will be slow there -- which is exactly the point of the benchmark.
GRID_SIZES = [128, 256, 512, 1024, 2048]


@dataclass
class OceanParams:
    """Physical parameters of the Phillips-spectrum ocean."""

    N: int = 512                 # grid resolution (NxN), power of two
    L: float = 250.0             # patch size in world units (meters)
    wind_speed: float = 30.0     # wind speed V (m/s) -> bigger waves
    wind_dir_deg: float = 45.0   # wind direction (degrees, CCW from +x)
    amplitude: float = 4.0       # global amplitude scale A of the spectrum
    choppiness: float = 1.3      # horizontal displacement strength (lambda)
    gravity: float = 9.81        # g (m/s^2)
    # Waves much smaller than this (meters) are suppressed -> tames aliasing.
    small_wave: float = 0.5
    time_scale: float = 1.0      # multiplies simulation time

    def copy(self) -> "OceanParams":
        return OceanParams(**self.__dict__)


@dataclass
class RenderParams:
    """Window / camera / shading parameters."""

    width: int = 1280
    height: int = 800
    target_fps: int = 60
    # World-space size the patch is drawn at (decoupled from physical L so the
    # camera framing stays put when you change resolution).
    world_size: float = 200.0
    # Vertical exaggeration of the displayed surface. The raw ifft height field
    # has a small RMS (~0.02), so this maps it to a few world units. Horizontal
    # choppy displacement uses a fraction of this (see renderer).
    height_scale: float = 220.0
    # Sun direction (will be normalised in the shader).
    sun_dir: tuple = (0.55, 0.8, 0.35)
    # Linear-ish colours (0..1).
    deep_color: tuple = (0.02, 0.10, 0.18)
    shallow_color: tuple = (0.10, 0.34, 0.42)
    sky_color: tuple = (0.55, 0.70, 0.92)
    sun_color: tuple = (1.0, 0.95, 0.82)


@dataclass
class AppConfig:
    ocean: OceanParams = field(default_factory=OceanParams)
    render: RenderParams = field(default_factory=RenderParams)
    # Backend to start with: "numpy" (CPU) or "cupy" (GPU). Falls back to numpy
    # automatically if CuPy / a CUDA GPU is unavailable.
    backend: str = "cupy"
