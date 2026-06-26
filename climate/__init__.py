"""
CAMulator climate-inference toolbox.

A small, self-contained set of scripts that roll a trained CAMulator checkpoint
forward for climate-length runs and write NetCDF output. The scripts are
meant to be run directly (``python Quick_Climate.py ...``) from this directory,
which is why this package does not eagerly import the heavy modules here (doing
so would require a GPU + the full ``credit`` stack just to ``import climate``).

Modules
-------
Quick_Climate                  : roll the model forward; per-6-hourly-step
                                 pred_*.nc, or --daily_mean / --monthly_mean
                                 time-averaged NetCDF
Model_State                    : state container + CAMulatorStepper (physics step)
WindPP                         : wind-artifact post-filter (called by Model_State)
Make_Climate_Initial_Conditions: build a new initial-condition tensor

See README.md for the end-to-end workflow and the asset manifest.
"""

__version__ = "0.3.0"
__all__ = [
    "Quick_Climate",
    "Model_State",
    "WindPP",
    "Make_Climate_Initial_Conditions",
]
