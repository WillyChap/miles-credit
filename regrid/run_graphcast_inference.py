"""End-to-end test: CAM6 h1 -> translator -> GraphCast -> +6h forecast.

This is the proof of life for the forward pipeline. It:

  1. Builds a GraphCast input bundle from two consecutive CAM6 h1 files
     (T-6h and T) via `regrid.graphcast_bundle.build_graphcast_input_bundle`.
  2. Loads the 0.25 deg / 37-level / mesh-2to6 ERA5 GraphCast checkpoint.
  3. Loads the per-level normalization stats.
  4. Runs a single 6-hour autoregressive step.
  5. Saves the predicted state to netCDF for inspection / use as the
     reverse-direction (GraphCast -> CAM6) source.

The first compiled run is slow (~minutes) because of JAX tracing; subsequent
calls reuse the compiled graph. We report both numbers.

Run:
    python -m regrid.run_graphcast_inference \\
        --h1-prev /path/to/...h1.1980-01-05-64800.nc \\
        --h1-t    /path/to/...h1.1980-01-06-00000.nc \\
        --out     /scratch/.../gc_forecast_1980-01-06-06000.nc
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from pathlib import Path

# Persistent XLA compilation cache: write compiled HLO to disk so a second
# invocation with the same input shapes skips the ~15 min compile entirely.
# Set BEFORE importing jax. The default cache lives on derecho scratch where
# every casper / derecho node can read it.
DEFAULT_JAX_CACHE = "/glade/derecho/scratch/wchapman/jax_compilation_cache"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", DEFAULT_JAX_CACHE)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

import haiku as hk
import jax
import numpy as np
import xarray as xr

from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast as gc_mod,
    normalization,
    rollout,
)

from regrid.graphcast_bundle import build_graphcast_input_bundle

# Repo defaults (absolute paths so the script is callable from anywhere).
REPO_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = (
    REPO_DIR / "graphcast" / "params" /
    "graphcast_params_GraphCast - ERA5 1979-2017 - resolution 0.25 - "
    "pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
)
STATS_DIR = REPO_DIR / "graphcast" / "stats"


def load_stats() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Load the three per-level normalization stats files into memory."""
    return (
        xr.load_dataset(STATS_DIR / "diffs_stddev_by_level.nc").compute(),
        xr.load_dataset(STATS_DIR / "mean_by_level.nc").compute(),
        xr.load_dataset(STATS_DIR / "stddev_by_level.nc").compute(),
    )


def load_checkpoint(path: Path = CHECKPOINT_PATH):
    with open(path, "rb") as f:
        ckpt = checkpoint.load(f, gc_mod.CheckPoint)
    return ckpt


def build_predictor_fn(
    model_config: gc_mod.ModelConfig,
    task_config: gc_mod.TaskConfig,
    diffs_stddev: xr.Dataset,
    mean: xr.Dataset,
    stddev: xr.Dataset,
):
    """Construct the haiku-transformed forward function used at inference."""

    def construct_wrapped(model_config, task_config):
        predictor = gc_mod.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev,
            mean_by_level=mean,
            stddev_by_level=stddev,
        )
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    return run_forward


def run_inference(
    h1_prev: Path, h1_t: Path, out_path: Path,
    bundle_only: bool = False,
) -> None:
    print("=" * 60)
    print(f"CAM6 -> GraphCast forward inference")
    print(f"  T-6h h1: {h1_prev.name}")
    print(f"  T   h1: {h1_t.name}")
    print(f"  output: {out_path}")
    print("=" * 60)

    # 1. Build input bundle from CAM6 h1 files.
    t0 = time.time()
    bundle = build_graphcast_input_bundle((h1_prev, h1_t))
    print(f"\nbundle built in {time.time() - t0:.1f}s")

    # 2. Load checkpoint + stats.
    t0 = time.time()
    ckpt = load_checkpoint()
    diffs_stddev, mean_by_level, stddev_by_level = load_stats()
    print(f"checkpoint + stats loaded in {time.time() - t0:.1f}s")
    print(f"  model: resolution={ckpt.model_config.resolution} "
          f"mesh_size={ckpt.model_config.mesh_size} "
          f"levels={len(ckpt.task_config.pressure_levels)}")

    # 3. Split the bundle into (inputs, targets, forcings).
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        bundle,
        input_variables=ckpt.task_config.input_variables,
        target_variables=ckpt.task_config.target_variables,
        forcing_variables=ckpt.task_config.forcing_variables,
        pressure_levels=ckpt.task_config.pressure_levels,
        input_duration=ckpt.task_config.input_duration,
        target_lead_times="6h",
    )
    print(f"\ninputs  : {dict(inputs.sizes)}")
    print(f"targets : {dict(targets.sizes)}")
    print(f"forcings: {dict(forcings.sizes)}")

    if bundle_only:
        bundle.to_netcdf(out_path)
        print(f"\nbundle saved to {out_path}; skipping inference (bundle_only=True)")
        return

    # 4. Build predictor + jit it.
    run_forward = build_predictor_fn(
        ckpt.model_config, ckpt.task_config,
        diffs_stddev, mean_by_level, stddev_by_level,
    )

    # Demo's wrappers use functools.partial with KWARGS (not positional
    # prepend) so the arg order on the underlying apply_fn is preserved.
    def with_configs(fn):
        return functools.partial(
            fn,
            model_config=ckpt.model_config,
            task_config=ckpt.task_config,
        )

    def with_params(fn):
        return functools.partial(fn, params=ckpt.params, state={})

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    run_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

    # 5. Inference.
    print(f"\nGraphCast inference...")
    print(f"  (first call ~10-15 min trace+compile;"
          f" subsequent calls reuse XLA cache at {os.environ['JAX_COMPILATION_CACHE_DIR']})")
    t0 = time.time()
    predictions = rollout.chunked_prediction(
        run_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets * np.nan,
        forcings=forcings,
    )
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s")
    print(f"\npredictions:")
    print(predictions)

    # 6. Save.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_netcdf(out_path)
    print(f"\nsaved +6h forecast to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1-prev", type=Path, required=True)
    ap.add_argument("--h1-t",    type=Path, required=True)
    ap.add_argument("--out",     type=Path, required=True)
    ap.add_argument("--bundle-only", action="store_true",
                    help="Just build and save the input bundle, no GC call.")
    args = ap.parse_args()
    run_inference(args.h1_prev, args.h1_t, args.out, bundle_only=args.bundle_only)


if __name__ == "__main__":
    main()
