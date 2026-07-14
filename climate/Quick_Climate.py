"""
Quick_Climate_V02.py
--------------------
Refactored CAMulator climate integration with clearer coupling interfaces.

Key improvements:
- Separated initialization from time-stepping
- Clear CAMulatorStepper class for coupling
- Documented state tensor structure
- Removed dead code
- Preserved async parallel I/O for performance
"""

import os
import time
import logging
import warnings
from pathlib import Path
import multiprocessing as mp
import argparse

from Model_State import initialize_camulator

# ---------- #
# Numerics
from datetime import datetime
import numpy as np

# ---------- #
import torch

# ---------- #
# credit
from credit.output import make_xarray, save_netcdf_increment

logger = logging.getLogger(__name__)
# Without this the root logger sits at WARNING and silently swallows the
# conservation-fixer diagnostics (mass/water/energy correction magnitudes).
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def add_init_noise(state: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """
    Add random noise to initial conditions for ensemble generation.

    Args:
        state: Initial state tensor
        noise_std: Standard deviation of Gaussian noise

    Returns:
        state_with_noise: Perturbed state
    """
    print(f"Adding initial condition noise (std={noise_std})")
    noise = torch.randn_like(state) * noise_std
    return state + noise


def parse_datetime_from_config(conf: dict) -> datetime:
    """
    Parse datetime from config, handling string, datetime, and cftime objects.

    Args:
        conf: Configuration dictionary

    Returns:
        init_dt: Python datetime object
    """
    raw_dt = conf["predict"]["start_datetime"]

    if isinstance(raw_dt, str):
        # Parse "YYYY-MM-DD HH:MM:SS" format
        return datetime.strptime(raw_dt, "%Y-%m-%d %H:%M:%S")
    elif isinstance(raw_dt, datetime):
        # Already a Python datetime
        return raw_dt
    else:
        # Assume it's a cftime object - convert to Python datetime
        # cftime objects have year, month, day, hour, minute, second attributes
        return datetime(raw_dt.year, raw_dt.month, raw_dt.day, raw_dt.hour, raw_dt.minute, raw_dt.second)


# ============================================================================
# INTEGRATION LOOP
# ============================================================================


# Averaging-period helpers. The averaging modes (--daily_mean / --monthly_mean)
# are a compact alternative to the default per-6-hourly-step output: they write
# one time-averaged NetCDF per day or per month, replacing the old Post_Process.py.
_PERIOD_FMT = {"daily": "%Y%m%d", "monthly": "%Y%m"}  # filename tag per period


def _bucket_key(dt, period):
    """Identity of the averaging bucket a timestamp falls in."""
    return (dt.year, dt.month, dt.day) if period == "daily" else (dt.year, dt.month)


def _flush_average(pool, accum, bucket_dt, init_str, period, metadata, conf, lats, lons):
    """Average accumulated predictions and save one NetCDF per day/month."""
    if not accum:
        return
    stacked = torch.stack(accum, dim=0).mean(dim=0)  # [1, C, 1, H, W] mean over bucket
    upper_air, single_level = make_xarray(stacked, bucket_dt, lats, lons, conf)
    file_tag = int(bucket_dt.strftime(_PERIOD_FMT[period]))  # YYYYMMDD or YYYYMM — human-readable
    pool.apply_async(save_netcdf_increment, (upper_air, single_level, init_str, file_tag, metadata, conf))
    print(f"  Saved {period} mean for {bucket_dt.strftime('%Y-%m-%d')}")


def run_climate_integration(pool: mp.Pool, context: dict, save_append: str = None, init_noise: float = None,
                            monthly_mean: bool = False, daily_mean: bool = False,
                            track_moisture: bool = False, no_wind_pp: bool = False):
    """
    Run the CAMulator climate integration loop.

    This function handles the time-stepping, output generation, and parallel I/O.
    The core physics stepping is delegated to the CAMulatorStepper class.

    Args:
        pool: Multiprocessing pool for async file I/O
        context: Dictionary from initialize_camulator()
        save_append: Optional subfolder name for outputs
        init_noise: Optional noise std to add to initial conditions

    Returns:
        flag_energy: Whether energy fixer was active (for diagnostics)
    """
    # Unpack context
    conf = context["conf"]
    stepper = context["stepper"]
    forcing_ds_norm = context["forcing_dataset"]
    static_forcing = context["static_forcing"]
    state = context["initial_state"]
    latlons = context["latlons"]
    metadata = context["metadata"]
    device = context["device"]

    # Optionally disable WindPP wind-artifact filtering (default: leave it ON).
    if no_wind_pp:
        stepper.enable_wind_filtering = False
        print("WindPP wind-artifact filtering DISABLED (--no_wind_pp)")
    print(f"Wind filtering active: {stepper.enable_wind_filtering}")

    # Update save location if append specified
    if save_append:
        base = conf["predict"].get("save_forecast")
        if not base:
            raise KeyError("'save_forecast' missing in config")
        conf["predict"]["save_forecast"] = str(Path(base).expanduser() / save_append)
        print(f"Saving outputs to: {conf['predict']['save_forecast']}")

    # Add noise to initial conditions if requested
    if init_noise is not None:
        state = add_init_noise(state, noise_std=init_noise)

    # Trace model for performance (optional but recommended)
    print("Tracing model with torch.jit...")
    # IMPORTANT: Initial state already contains forcing for first timestep
    # So we trace with the initial state shape as-is (DO NOT add forcing channels)
    dummy_input = torch.zeros_like(state)
    traced_model = torch.jit.trace(stepper.model, dummy_input)
    stepper.model = traced_model
    print(f"Model traced with input shape: {dummy_input.shape}")

    # Setup for time-stepping
    df_vars = conf["data"]["dynamic_forcing_variables"]
    num_ts = conf["predict"]["timesteps_fast_climate"]
    lead_time_periods = conf["data"]["lead_time_periods"]
    chunk_size = conf["data"].get("forcing_chunk_size", 32)

    # Get forcing data subset
    dynamic_ds = forcing_ds_norm[df_vars]

    # Convert start_datetime to whatever type the time index uses (cftime or datetime)
    start_datetime_raw = conf["predict"]["start_datetime"]
    time_index = dynamic_ds.indexes["time"]

    if isinstance(start_datetime_raw, str):
        from datetime import datetime as _dt
        dt = _dt.strptime(start_datetime_raw, "%Y-%m-%d %H:%M:%S")
    else:
        dt = start_datetime_raw

    # Find index by direct year/month/day/hour comparison — avoids any
    # cftime has_year_zero / calendar-type construction mismatch.
    import numpy as _np
    t_arr = _np.array(time_index)
    months = _np.array([t.month for t in t_arr])
    days   = _np.array([t.day   for t in t_arr])
    hours  = _np.array([t.hour  for t in t_arr])

    # Try exact year+month+day+hour first; fall back to month+day+hour for cyclic files
    years  = _np.array([t.year  for t in t_arr])
    mask_exact = (years == dt.year) & (months == dt.month) & (days == dt.day) & (hours == dt.hour)
    mask_cyclic = (months == dt.month) & (days == dt.day) & (hours == dt.hour)

    hits = _np.where(mask_exact)[0]
    if len(hits) == 0:
        hits = _np.where(mask_cyclic)[0]
        if len(hits) > 0:
            print(f"  Cyclic forcing: matched {start_datetime_raw!r} by month/day/hour at index {hits[0]}")
    assert len(hits) > 0, f"start_datetime {start_datetime_raw!r} not found in forcing time index"
    start_ix = int(hits[0])
    print(f"Starting integration at time index: {start_ix}")

    # Now convert to Python datetime for output formatting (if it's a string or cftime)
    init_dt = parse_datetime_from_config(conf)
    init_str = init_dt.strftime("%Y-%m-%dT%HZ")

    # ========================================================================
    # MAIN TIME-STEPPING LOOP
    # ========================================================================

    # Pick the averaging mode (default: write every 6-hourly step).
    if monthly_mean and daily_mean:
        raise ValueError("Choose only one of monthly_mean / daily_mean")
    avg_period = "monthly" if monthly_mean else ("daily" if daily_mean else None)

    print("Starting time-stepping loop...")
    print(f"Output mode: {avg_period + ' means' if avg_period else 'every 6-hourly step'}")
    forecast_hour = 1
    timestep_counter = 0
    from datetime import timedelta
    import cftime
    # No-leap (365-day) simulation clock to match the CESM / forcing calendar.
    # A real datetime clock would insert Feb 29 and drift ~1 day every 4 yr away
    # from the no-leap forcing (e.g. a 35-yr run ends ~9 days early). DatetimeNoLeap
    # + timedelta skips Feb 29, so output stamps stay aligned with forcing time[i].
    sim_dt = cftime.DatetimeNoLeap(init_dt.year, init_dt.month, init_dt.day,
                                   init_dt.hour, init_dt.minute, init_dt.second)
    # simulation clock — advances independently of cyclic forcing, no-leap calendar

    # Averaging accumulation state (used when avg_period is "daily" or "monthly")
    avg_accum = []
    current_bucket = None
    bucket_start_dt = None

    # Moisture tracking state (Qtot channels 96-127, q_inds from post_conf)
    q_lo, q_hi = 96, 128  # 32 levels of Qtot
    moisture_log = []  # list of (step, datetime_str, qtot_mean)

    n_forcing = len(dynamic_ds.time)  # size of cyclic forcing file (e.g. 1460 for 1-yr)

    for block_start in range(0, num_ts, chunk_size):
        block_end = min(block_start + chunk_size, num_ts)

        # Cyclic wrap: map simulation steps to forcing indices
        forcing_indices = [(start_ix + block_start + i) % n_forcing for i in range(block_end - block_start)]

        # Load chunk of dynamic forcing data (with cyclic wrap)
        ds_slice = dynamic_ds.isel(time=forcing_indices).load()
        ds_slice_times = ds_slice["time"].values

        # Stack forcing variables into tensor [time, vars, lat, lon]
        arr_list = [ds_slice[var].values for var in dynamic_ds.data_vars]
        arr = np.stack(arr_list, axis=1)

        # Transfer to GPU once per chunk
        cpu_tensor = torch.from_numpy(arr).unsqueeze(2).pin_memory()
        gpu_forcing_chunk = cpu_tensor.to(device, non_blocking=True)

        # Step through each time in the chunk
        for t in range(gpu_forcing_chunk.shape[0]):
            # Use simulation clock (not forcing file time) so cyclic forcing
            # doesn't reset the year counter after 1 year
            utc_datetime = sim_dt

            if (timestep_counter + 1) % 20 == 0:
                msg = f"Model step: {timestep_counter + 1:05}, time: {utc_datetime}"
                if track_moisture and moisture_log:
                    msg += f"  |  Qtot_total={moisture_log[-1][2]:.3e}"
                print(msg)

            dynamic_forcing_t = gpu_forcing_chunk[t].unsqueeze(0)

            # ================================================================
            # CORE PHYSICS STEP
            # This matches the original Quick_Climate.py logic exactly:
            # - First step (timestep_counter=0): state already has forcing, run model as-is
            # - Subsequent steps: add forcing to state, then run model
            # ================================================================

            if timestep_counter != 0:
                # Build forcing from dynamic + static
                model_input = stepper.state_manager.build_input_with_forcing(state, dynamic_forcing_t, static_forcing)
            else:
                # First iteration: initial state already contains forcing
                model_input = state

            # Run model
            with torch.no_grad():
                prediction = stepper.model(model_input.float())

            # Apply post-processing
            prediction = stepper._apply_postprocessing(prediction, model_input)

            # Track global mean Qtot (normalized units — good for drift detection)
            if track_moisture:
                qtot_total = prediction[0, q_lo:q_hi, 0, :, :].sum().item()
                moisture_log.append((timestep_counter + 1, str(utc_datetime), qtot_total))

            timestep_counter += 1

            # ================================================================
            # OUTPUT GENERATION (runs in parallel via multiprocessing)
            # save_netcdf_increment handles inverse_transform internally via
            # climate_rescale_output: True in the config
            # ================================================================

            if avg_period:
                # Accumulate predictions; flush to disk when the day/month rolls over
                pred_cpu = prediction.cpu()
                bucket = _bucket_key(utc_datetime, avg_period)
                if current_bucket is None:
                    current_bucket = bucket
                    bucket_start_dt = utc_datetime
                elif bucket != current_bucket:
                    _flush_average(
                        pool, avg_accum, bucket_start_dt, init_str,
                        avg_period, metadata, conf,
                        latlons.latitude.values, latlons.longitude.values,
                    )
                    avg_accum = []
                    current_bucket = bucket
                    bucket_start_dt = utc_datetime
                avg_accum.append(pred_cpu)
            else:
                # Convert prediction to xarray (fast, on CPU)
                upper_air, single_level = make_xarray(
                    prediction.cpu(), utc_datetime, latlons.latitude.values, latlons.longitude.values, conf
                )
                # Async save to NetCDF (runs in background pool)
                pool.apply_async(
                    save_netcdf_increment,
                    (upper_air, single_level, init_str, lead_time_periods * forecast_hour, metadata, conf),
                )

            # ================================================================
            # SHIFT STATE FORWARD FOR NEXT TIMESTEP
            # ================================================================

            state = stepper.state_manager.shift_state_forward(state, prediction)
            forecast_hour += 1
            sim_dt += timedelta(hours=6)

    # Flush any remaining accumulated steps (final partial/full day or month)
    if avg_period and avg_accum:
        _flush_average(
            pool, avg_accum, bucket_start_dt, init_str,
            avg_period, metadata, conf,
            latlons.latitude.values, latlons.longitude.values,
        )

    # Save moisture tracking CSV
    if track_moisture and moisture_log:
        import csv
        save_dir = conf["predict"].get("save_forecast", ".")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        csv_path = str(Path(save_dir) / "moisture_tracking.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "datetime", "qtot_global_total_norm"])
            writer.writerows(moisture_log)
        print(f"Moisture tracking saved to: {csv_path}")

    print("Time-stepping complete. Waiting for I/O to finish...")
    time.sleep(30)  # Allow async writes to complete

    # Report BOTH energy fixers. CAMulator is trained with the up/down-flux variant
    # (global_energy_fixer_updown), so reporting only `flag_energy` hides whether the
    # constraint the model was trained under is actually being applied.
    energy_active = stepper.flag_energy or stepper.flag_energy_updown
    print(
        f"Integration finished. Energy fixer active: {energy_active} "
        f"(net={stepper.flag_energy}, updown={stepper.flag_energy_updown})"
    )
    return energy_active


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Command-line interface for running CAMulator climate integrations."""
    parser = argparse.ArgumentParser(
        description="Run CAMulator climate integration with clean coupling interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python Quick_Climate_V02.py \\
        --config ./be21_coupled-v2025.2.0_small.yml \\
        --model_name checkpoint.pt00091.pt \\
        --save_append run_future_00091 \\
        """,
    )

    parser.add_argument("--config", type=str, required=True, help="Path to model configuration YAML file")
    parser.add_argument(
        "--model_name", type=str, default=None, help="Optional model checkpoint name (e.g., checkpoint.pt00091.pt)"
    )
    parser.add_argument("--save_append", type=str, default=None, help="Append subfolder name to output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument(
        "--init_noise", type=float, default=None, help="Add Gaussian noise to initial conditions (for ensembles)"
    )
    parser.add_argument(
        "--monthly_mean", action="store_true", default=False,
        help="Save monthly means instead of every 6-hourly step (much smaller output)"
    )
    parser.add_argument(
        "--daily_mean", action="store_true", default=False,
        help="Save daily means instead of every 6-hourly step (smaller output)"
    )
    parser.add_argument(
        "--track_moisture", action="store_true", default=False,
        help="Track global mean column Qtot each step and save to moisture_tracking.csv"
    )
    parser.add_argument(
        "--no_wind_pp", action="store_true", default=False,
        help="Disable WindPP wind-artifact filtering (sets stepper.enable_wind_filtering=False)"
    )

    # Deprecated arguments (kept for backwards compatibility but unused)
    parser.add_argument(
        "--input_shape", type=int, nargs="+", default=None, help="[DEPRECATED] Input shape is now derived from config"
    )
    parser.add_argument(
        "--forcing_shape",
        type=int,
        nargs="+",
        default=None,
        help="[DEPRECATED] Forcing shape is now derived from config",
    )
    parser.add_argument(
        "--output_shape", type=int, nargs="+", default=None, help="[DEPRECATED] Output shape is now derived from config"
    )

    args = parser.parse_args()

    if args.input_shape or args.forcing_shape or args.output_shape:
        print("WARNING: --input_shape, --forcing_shape, --output_shape are deprecated.")
        print("         These are now automatically derived from the config file.")

    start_time = time.time()

    # Initialize CAMulator
    context = initialize_camulator(config_path=args.config, model_name=args.model_name, device=args.device)

    # Run integration with parallel I/O
    num_cpus = 8
    with mp.Pool(num_cpus) as pool:
        flag_energy = run_climate_integration(
            pool=pool, context=context, save_append=args.save_append, init_noise=args.init_noise,
            monthly_mean=args.monthly_mean, daily_mean=args.daily_mean,
            track_moisture=args.track_moisture, no_wind_pp=args.no_wind_pp,
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'=' * 60}")
    print("Run completed successfully!")
    print(f"Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print(f"Outputs saved to: {context['conf']['predict']['save_forecast']}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
