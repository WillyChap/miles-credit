"""Vectorized log-pressure interpolation for atmospheric columns.

Two directions:
  - hybrid -> pressure (CAM6 -> GraphCast): source p depends on PS, target p is fixed.
  - pressure -> hybrid (GraphCast -> CAM6): target p depends on PS, source p is fixed.

The reference implementation uses a single broadcast comparison to find brackets
in both directions; for 32 source x 37 target x 192*288 cols on the f09 grid
that allocates ~65 MB transiently, which is fine for offline reference. The
JAX runtime version (fused_translate.py) uses a per-column search that does
not materialize the broadcast matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VerticalInterpResult:
    """Output of a vectorized log-p interpolation.

    Attributes
    ----------
    field : np.ndarray
        Interpolated values, shape (nlev_tgt, ...). Where above_top or
        below_ground is True, `field` holds an extrapolated value (linear in
        log-p outside the source bracket) so downstream code can either keep
        it, replace it, or smoothly blend.
    above_top : np.ndarray of bool
        True where target pressure is LESS than source-top pressure (i.e.,
        target level is above the source TOA). Shape (nlev_tgt, ...).
    below_ground : np.ndarray of bool
        True where target pressure is GREATER than source-bottom pressure
        (i.e., target level is below the source surface). Shape (nlev_tgt, ...).
    """

    field: np.ndarray
    above_top: np.ndarray
    below_ground: np.ndarray


def interp_log_p(
    field: np.ndarray,
    p_src: np.ndarray,
    p_tgt: np.ndarray,
) -> VerticalInterpResult:
    """Linear interpolation in log(p), vectorized over columns.

    Parameters
    ----------
    field : np.ndarray
        Source field, shape (nlev_src, ...). Trailing axes are columns.
    p_src : np.ndarray
        Source pressures in Pa, shape (nlev_src, ...). Must be strictly
        increasing along axis 0 (top to bottom in pressure).
    p_tgt : np.ndarray
        Target pressures in Pa, shape (nlev_tgt,). Must be strictly increasing.

    Returns
    -------
    VerticalInterpResult

    Notes
    -----
    Within bracket: linear interpolation in log(p), exact on linear-in-log-p
    functions (including constants).
    Outside bracket: linear extrapolation in log(p) using the nearest two
    source levels. The above_top / below_ground masks let downstream code
    apply variable-specific rules (ERA5 T-bridge, hold-lowest, etc.).
    """
    p_src = np.asarray(p_src, dtype=np.float64)
    p_tgt = np.asarray(p_tgt, dtype=np.float64)
    field = np.asarray(field)

    assert p_src.shape[0] == field.shape[0], (
        f"p_src nlev {p_src.shape[0]} != field nlev {field.shape[0]}"
    )
    assert p_tgt.ndim == 1, "p_tgt must be 1-D (fixed target levels)"
    nlev_src = p_src.shape[0]
    nlev_tgt = p_tgt.size

    log_src = np.log(p_src)              # (nlev_src, ...)
    log_tgt = np.log(p_tgt)              # (nlev_tgt,)

    # For each target level l, k0[l, ...] = largest src index with log_src[k0] <= log_tgt[l].
    # Use broadcast comparison: cmp[l, k, ...] = log_src[k, ...] <= log_tgt[l].
    # k0 = cmp.sum(axis=1) - 1, clipped to [0, nlev_src-2].
    cmp = log_src[None, ...] <= log_tgt[(slice(None),) + (None,) * log_src.ndim]
    k0 = cmp.sum(axis=1) - 1                              # (nlev_tgt, ...)
    k0 = np.clip(k0, 0, nlev_src - 2)
    k1 = k0 + 1

    # Gather source log-p and field values at the bracket levels.
    # Use take_along_axis to avoid building fancy-index meshes for the column axes.
    log_below = np.take_along_axis(log_src, k0, axis=0)   # (nlev_tgt, ...)
    log_above = np.take_along_axis(log_src, k1, axis=0)
    f_below = np.take_along_axis(field, k0, axis=0)
    f_above = np.take_along_axis(field, k1, axis=0)

    # Linear weight in log-p, with safe denominator (p_src strictly increasing
    # implies log_above > log_below, so no zero divide).
    denom = log_above - log_below
    log_tgt_b = log_tgt[(slice(None),) + (None,) * (log_src.ndim - 1)]
    w = (log_tgt_b - log_below) / denom

    out = (1.0 - w) * f_below + w * f_above

    # Above-top mask: log_tgt < log_src[0].
    above_top = log_tgt_b < log_src[0:1, ...]
    # Below-ground mask: log_tgt > log_src[-1].
    below_ground = log_tgt_b > log_src[-1:, ...]

    # When above_top or below_ground, `out` is now an extrapolation along the
    # outermost bracket; that's the right starting point for variable-specific
    # rules.

    return VerticalInterpResult(field=out, above_top=above_top, below_ground=below_ground)


def interp_log_p_to_hybrid(
    field_plev: np.ndarray,
    p_src_plev: np.ndarray,
    p_tgt_hybrid: np.ndarray,
) -> VerticalInterpResult:
    """Pressure-level source -> hybrid-level target.

    Inverse of `interp_log_p` in role but same math: the only difference is
    that the source pressures here are fixed (1-D, common to all columns) and
    the target pressures depend on PS (shape matches columns). For convenience
    we broadcast p_src_plev across columns so the same kernel can be reused.

    Parameters
    ----------
    field_plev : np.ndarray
        (nlev_src,) + col_shape -- but with fixed-per-column pressure levels.
    p_src_plev : np.ndarray
        (nlev_src,) Pa, strictly increasing.
    p_tgt_hybrid : np.ndarray
        (nlev_tgt,) + col_shape Pa, strictly increasing along axis 0 per column.

    Returns
    -------
    VerticalInterpResult with field of shape (nlev_tgt,) + col_shape.
    """
    p_src_plev = np.asarray(p_src_plev, dtype=np.float64)
    p_tgt_hybrid = np.asarray(p_tgt_hybrid, dtype=np.float64)
    field_plev = np.asarray(field_plev)

    assert p_src_plev.ndim == 1, "p_src_plev (pressure levels) must be 1-D"
    col_shape = p_tgt_hybrid.shape[1:]
    assert field_plev.shape == (p_src_plev.size,) + col_shape, (
        f"field_plev shape {field_plev.shape} must be {(p_src_plev.size,) + col_shape}"
    )

    log_src = np.log(p_src_plev)                          # (nlev_src,)
    log_tgt = np.log(p_tgt_hybrid)                        # (nlev_tgt, *col)

    # For each (target level, column), k0 = #(log_src <= log_tgt) - 1, clipped.
    # log_src has 1 axis, log_tgt has 1 + ncol axes. Build broadcast:
    log_src_b = log_src.reshape((log_src.size,) + (1,) * len(col_shape))   # (nlev_src, 1, 1, ...)
    cmp = log_src_b[None, ...] <= log_tgt[:, None, ...]    # (nlev_tgt, nlev_src, *col)
    nlev_src = log_src.size
    k0 = cmp.sum(axis=1) - 1                               # (nlev_tgt, *col)
    k0 = np.clip(k0, 0, nlev_src - 2)
    k1 = k0 + 1

    # log_src is 1-D, so log_src[k0] is correct (shape matches k0).
    log_below = log_src[k0]                                # (nlev_tgt, *col)
    log_above = log_src[k1]
    # field_plev is (nlev_src, *col) — bare fancy indexing would broadcast
    # to (k0.shape) + field_plev.shape[1:] = (nlev_tgt, *col, *col) and
    # blow up memory. Use take_along_axis so the index array aligns with
    # field_plev's axis-0 only.
    f_below = np.take_along_axis(field_plev, k0, axis=0)   # (nlev_tgt, *col)
    f_above = np.take_along_axis(field_plev, k1, axis=0)

    w = (log_tgt - log_below) / (log_above - log_below)
    out = (1.0 - w) * f_below + w * f_above

    above_top = log_tgt < log_src[0]
    below_ground = log_tgt > log_src[-1]

    return VerticalInterpResult(field=out, above_top=above_top, below_ground=below_ground)


__all__ = [
    "VerticalInterpResult",
    "interp_log_p",
    "interp_log_p_to_hybrid",
]
