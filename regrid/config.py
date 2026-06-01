"""Single source of truth for the CAM6 <-> GraphCast translation layer.

Everything that is conceptually fixed (variable names, units, pressure levels,
direction conventions) lives here. Code in other modules imports these constants
rather than redefining them. If a name changes, it changes in one place.

Confirmed against:
  - CAM6 h1 file:
      /glade/derecho/scratch/wchapman/g.e21.SUMO_CAM6_v03/run/
      g.e21.SUMO_CAM6_v03.cam.h1.1980-01-01-21600.nc
  - GraphCast checkpoint:
      camulator_sumo/graphcast/params/
      "graphcast_params_GraphCast - ERA5 1979-2017 - resolution 0.25 - "
      "pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
  - graphcast/graphcast/graphcast.py TaskConfig `TASK`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# -----------------------------------------------------------------------------
# Reference h1 files (persistent disk)
# -----------------------------------------------------------------------------
# Used by graphcast_server.py / smoke_test / launch_gc_test.sh to:
#   - precompute the CAM6 <-> GraphCast bilinear stencil
#   - build a 2-frame warmup bundle so JIT compile completes BEFORE the server
#     signals READY
#
# These files live under /glade/work (persistent), copies of the first two
# h1 outputs from the validated SUMO_CAM6_v03 5-day run. They carry all the
# fincl2 fields GraphCast needs (OMEGA, Z3, TREFHT, PSL, U10/UBOT/VBOT,
# PRECT, PHIS, LANDFRAC). If the fincl2 set changes, refresh them.

_REGRID_DIR = Path(__file__).resolve().parent
REFERENCE_DATA_DIR = _REGRID_DIR / "reference_data"
REFERENCE_H1_PREV = REFERENCE_DATA_DIR / "reference_h1_prev.nc"  # T-6h
REFERENCE_H1_T    = REFERENCE_DATA_DIR / "reference_h1_t.nc"     # T

# -----------------------------------------------------------------------------
# Pressure levels
# -----------------------------------------------------------------------------

# ERA5 37-level set, ASCENDING (TOA -> surface) in hPa. Matches
# PRESSURE_LEVELS_ERA5_37 in graphcast/graphcast/graphcast.py.
PRESSURE_LEVELS_HPA = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700,
    750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000,
)
NUM_PLEV = len(PRESSURE_LEVELS_HPA)  # 37

# Internally everything is Pa. Convert hPa -> Pa once at config load.
PRESSURE_LEVELS_PA = tuple(p * 100.0 for p in PRESSURE_LEVELS_HPA)


# -----------------------------------------------------------------------------
# CAM6 grid + hybrid coordinate
# -----------------------------------------------------------------------------

CAM6_NLAT = 192       # FV f09 (~0.94 deg)
CAM6_NLON = 288       # FV f09 (~1.25 deg)
CAM6_NLEV = 32        # hybrid midpoints
CAM6_NILEV = 33       # hybrid interfaces
CAM6_P0_PA = 100000.0 # reference pressure
# CAM6 vertical ordering: `lev` has positive=down -> index 0 is TOA, index 31 is surface.
CAM6_LEV_POSITIVE_DOWN = True


# -----------------------------------------------------------------------------
# GraphCast grid
# -----------------------------------------------------------------------------

GC_NLAT = 721         # 0.25 deg, includes both poles (-90, ..., +90)
GC_NLON = 1440        # 0.25 deg, 0 .. 359.75


# -----------------------------------------------------------------------------
# Variable mapping: CAM6 history name -> GraphCast input name
# -----------------------------------------------------------------------------

# Atmospheric (per-level): CAM6 name -> GraphCast TASK input name.
ATMO_VAR_MAP = {
    "T":     "temperature",            # K
    "Z3":    "geopotential",           # CAM6: m (height); ERA5/GC: m^2/s^2. Multiply by g.
    "U":     "u_component_of_wind",    # m/s
    "V":     "v_component_of_wind",    # m/s
    "OMEGA": "vertical_velocity",      # Pa/s
    "Q":     "specific_humidity",      # kg/kg
}

# Surface (2D): CAM6 name -> GraphCast TASK input name.
# 10m wind components: CAM6 has only U10 magnitude; the translator
# reconstructs components from UBOT/VBOT direction. See coupling.py.
SURFACE_VAR_MAP = {
    "TREFHT": "2m_temperature",                  # K
    "PSL":    "mean_sea_level_pressure",         # Pa
    # 10m_u_component_of_wind / 10m_v_component_of_wind: synthesized
    "PRECT":  "total_precipitation_6hr",         # CAM6: m/s avg over 6h; GC/ERA5: m. Multiply by 21600.
}

# Static (time-invariant per case). PHIS is m^2/s^2 in CAM and m^2/s^2 in ERA5 -> direct.
STATIC_VAR_MAP = {
    "PHIS":     "geopotential_at_surface",  # m^2/s^2
    "LANDFRAC": "land_sea_mask",            # fraction (0..1)
}


# -----------------------------------------------------------------------------
# Below-ground / extrapolation policy
# -----------------------------------------------------------------------------
# Variable-specific rules for pressure levels with p_gc > ps_cam.
# Implemented in below_ground_era5.py — this enum just labels the policy.

@dataclass(frozen=True)
class BelowGroundPolicy:
    name: str
    # If True, the underground value is clipped to be >= 0 after extrapolation.
    clip_nonneg: bool = False


BG_POLICY = {
    "u_component_of_wind":  BelowGroundPolicy("hold_lowest"),
    "v_component_of_wind":  BelowGroundPolicy("hold_lowest"),
    "specific_humidity":    BelowGroundPolicy("hold_lowest", clip_nonneg=True),
    "vertical_velocity":    BelowGroundPolicy("hold_lowest"),
    "temperature":          BelowGroundPolicy("era5_t_bridge"),
    "geopotential":         BelowGroundPolicy("era5_z_hydrostatic"),
}


# -----------------------------------------------------------------------------
# Physical constants (SI)
# -----------------------------------------------------------------------------

GRAVITY = 9.80616             # m/s^2 (CESM cam6 cnst_grav)
RD = 287.04                   # J/kg/K dry air gas constant
RV = 461.50                   # J/kg/K water vapor gas constant
EPS_MV_OVER_MA = RD / RV      # ~0.622
T_LAPSE = 0.0065              # K/m, ICAO standard lapse used for T_bridge below-ground
T_LAPSE_ZERO_AT_2K = 2000.0   # m, lapse rate tapered above this height (Trenberth, FULL-POS-like)


# -----------------------------------------------------------------------------
# Translator runtime flags
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TranslatorConfig:
    """Knobs for runtime translation. Defaults are validated for offline reference."""

    # When True, the reference NumPy translator runs alongside the fused JAX one
    # and asserts agreement within tolerance. Off for production runtime.
    cross_check_reference: bool = False
    reference_rtol: float = 1e-4
    reference_atol: float = 1e-3

    # Below-ground T-bridge tapering: if True, taper the standard-atmosphere
    # lapse rate to zero in the lowest ~2km, matching ECMWF FULL-POS behavior
    # over high topography. If False, use a constant 6.5 K/km. The taper costs
    # one extra division per underground point.
    t_bridge_taper_lapse: bool = True

    # When CAM6 surface is BELOW the GraphCast 1000 hPa level (lowland/storm),
    # the GC->CAM6 reverse interp extrapolates downward in log(p). This flag
    # caps extrapolation distance; beyond it the lowest GC level is held.
    gc_to_cam_max_extrap_dlogp: float = 0.05  # ~5% in p


DEFAULT_CONFIG = TranslatorConfig()


__all__ = [
    "PRESSURE_LEVELS_HPA",
    "PRESSURE_LEVELS_PA",
    "NUM_PLEV",
    "CAM6_NLAT",
    "CAM6_NLON",
    "CAM6_NLEV",
    "CAM6_NILEV",
    "CAM6_P0_PA",
    "CAM6_LEV_POSITIVE_DOWN",
    "GC_NLAT",
    "GC_NLON",
    "ATMO_VAR_MAP",
    "SURFACE_VAR_MAP",
    "STATIC_VAR_MAP",
    "BG_POLICY",
    "BelowGroundPolicy",
    "GRAVITY",
    "RD",
    "RV",
    "EPS_MV_OVER_MA",
    "T_LAPSE",
    "T_LAPSE_ZERO_AT_2K",
    "TranslatorConfig",
    "DEFAULT_CONFIG",
]
