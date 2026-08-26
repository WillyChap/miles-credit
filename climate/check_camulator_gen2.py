#!/usr/bin/env python
"""Pre-flight checks for a CAMulator gen2 config.

Every failure mode this checks for is one that otherwise passes silently: the run starts, the
log looks healthy, and the physics is wrong. Run it before a rollout or a fine-tune.

    python check_camulator_gen2.py camulator_gen2.yml [--train]

--train additionally checks the settings that only matter when training.
Exit status is 0 when every check passes, 1 otherwise.
"""

import argparse
import os
import sys

import numpy as np

import yaml

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "
_failed = []
_warned = []


def check(name, condition, detail=""):
    print(f"[{OK if condition else BAD}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        _failed.append(name)


def warn(name, condition, detail=""):
    print(f"[{OK if condition else WARN}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        _warned.append(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--train", action="store_true", help="also check training-only settings")
    args = ap.parse_args()

    import credit
    from credit.parser import credit_main_parser

    print(f"config : {args.config}")
    print(f"credit : {credit.__file__}\n")

    raw = yaml.safe_load(open(args.config))
    _pcraw = (raw.get("model") or {}).get("post_conf") or {}
    check(
        "not both energy fixers active",
        not (
            _pcraw.get("global_energy_fixer", {}).get("activate")
            and _pcraw.get("global_energy_fixer_updown", {}).get("activate")
        ),
        "both enforce the same budget by rescaling T -- they would double-correct",
    )

    try:
        conf = credit_main_parser(dict(raw), parse_training=False, parse_predict=True)
    except Exception as exc:  # noqa: BLE001 - report the parse failure as a failed check
        check("config parses", False, f"{type(exc).__name__}: {exc}")
        print("\nFAILED: the config does not parse; nothing else could be checked")
        return 1
    pc = conf["model"]["post_conf"]
    vin, vout = pc["varname_input"], pc["varname_output"]

    print("-- channel map --")
    check("input tensor is 136 channels", len(vin) == 136, f"got {len(vin)}")
    check("output tensor is 147 channels", len(vout) == 147, f"got {len(vout)}")
    check("static_first is True", conf["data"].get("static_first") is True)

    print("\n-- energy up/down fixer --")
    ud = pc.get("global_energy_fixer_updown", {})
    check("up/down energy fixer is active", ud.get("activate") is True)
    check(
        "net-flux energy fixer is OFF",
        pc.get("global_energy_fixer", {}).get("activate") is not True,
        "both active would double-correct T",
    )
    solin = ud.get("TOA_forcing_solar_ind")
    check(
        "SOLIN index points at SOLIN in the INPUT tensor",
        solin is not None and 0 <= solin < len(vin) and vin[solin] == "SOLIN",
        f"index {solin} -> {vin[solin] if solin is not None and solin < len(vin) else '?'}",
    )
    for key, expect in [
        ("TOA_up_solar_ind", "FSUTOA"),
        ("TOA_up_OLR_ind", "FLUT"),
        ("surf_down_solar_ind", "FSDS_J"),
        ("surf_up_solar_ind", "FSUS"),
        ("surf_down_LW_ind", "FLDS_J"),
        ("surf_up_LW_ind", "FLUS"),
        ("surf_SH_ind", "SHFLX"),
        ("surf_LH_ind", "LHFLX"),
    ]:
        i = ud.get(key)
        got = vout[i] if i is not None and 0 <= i < len(vout) else "?"
        check(f"{key} -> {expect}", got == expect, f"index {i} -> {got}")

    print("\n-- scaling --")
    import xarray as xr

    # PS is the only 2-D variable whose statistics are a field. Any scaler that stores one
    # value per channel replaces it with the global mean, silently.
    ps_key = None
    for src, cfg in (conf.get("data", {}).get("source") or {}).items():
        if "PS" in ((cfg.get("variables", {}).get("prognostic") or {}).get("vars_2D") or []):
            ps_key = f"{src}/prognostic/2d/PS"
    for section in ("preblocks", "postblocks"):
        for name, blk in ((raw.get(section) or {}).get("per_step") or {}).items():
            if not isinstance(blk, dict) or blk.get("type") != "bridgescaler_transform":
                continue
            spatial = (blk.get("args") or {}).get("spatial_variables") or []
            check(
                f"{section}.{name}: PS is in spatial_variables",
                ps_key in spatial,
                f"expected {ps_key!r}, got {spatial}",
            )
            path = (blk.get("args") or {}).get("scaler_path")
            if path and os.path.exists(os.path.expandvars(path)):
                import base64
                import json as _json

                entry = None
                for dtype_block in _json.load(open(os.path.expandvars(path))).values():
                    for srcblk in dtype_block.values():
                        if ps_key in srcblk:
                            entry = _json.loads(srcblk[ps_key])
                            break
                if entry is None:
                    # Not found is a failure, not a skip: a scaler that does not cover PS
                    # under this exact key leaves PS unscaled or keyed to another source name.
                    check(
                        f"{section}.{name}: scaler JSON contains PS",
                        False,
                        f"{ps_key!r} not present in {path} -- wrong source name, or PS was omitted",
                    )
                else:
                    o = entry["mean_x_"]
                    n = np.frombuffer(base64.b64decode(o["__numpy__"]), dtype=np.dtype(o["dtype"])).size
                    check(
                        f"{section}.{name}: stored PS scaler is a field, not a scalar",
                        n > 1,
                        f"PS mean_x_ holds {n} value(s); expected one per gridpoint",
                    )

    mean_path = conf["data"].get("mean_path")
    if mean_path and os.path.exists(mean_path):
        ps = xr.open_dataset(mean_path).get("PS")
        check(
            "PS statistics are a 2-D (lat, lon) field",
            ps is not None and ps.ndim == 2,
            f"shape {tuple(ps.shape) if ps is not None else 'missing'}",
        )

    # The gen2 metric layer scores y_processed against y_target_processed, built by running the
    # same chain on the flat target. Supply only the prediction half and metrics silently compare
    # physical predictions against normalized targets -- the loss still looks fine.
    pb = (raw.get("postblocks") or {}).get("per_step") or {}
    if pb:

        def _keys_for(target):
            out = []
            for name, blk in pb.items():
                if not isinstance(blk, dict):
                    continue
                a = blk.get("args") or {}
                key = a.get("out_key") or a.get("key") or "y_processed"
                if key == target:
                    out.append((name, blk.get("type")))
            return out

        pred = _keys_for("y_processed")
        tgt = _keys_for("y_target_processed")

        def has(pairs, t):
            return any(ty == t for _, ty in pairs)

        check(
            "postblocks: prediction chain has reconstruct + inverse scaler",
            has(pred, "reconstruct") and has(pred, "bridgescaler_transform"),
            str(pred),
        )
        check(
            "postblocks: target chain has reconstruct + inverse scaler",
            has(tgt, "reconstruct") and has(tgt, "bridgescaler_transform"),
            f"{tgt} -- without a scaler on y_target_processed the metrics compare physical "
            f"predictions against normalized targets (valid_rmse ~1e6)",
        )

    print("\n-- fixers construct --")
    from credit.postblock.gen1 import (
        GlobalEnergyFixerUpDown,
        GlobalMassFixer,
        GlobalWaterFixer,
        TracerFixer,
    )

    for name, cls in [
        ("tracer_fixer", TracerFixer),
        ("global_mass_fixer", GlobalMassFixer),
        ("global_water_fixer", GlobalWaterFixer),
        ("global_energy_fixer_updown", GlobalEnergyFixerUpDown),
    ]:
        if not pc.get(name, {}).get("activate"):
            continue
        try:
            cls(pc)
            check(f"{name} constructs", True)
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            check(f"{name} constructs", False, f"{type(exc).__name__}: {exc}")

    print("\n-- referenced files exist --")
    for section, key in [
        ("data", "mean_path"),
        ("data", "std_path"),
        ("data", "save_loc_static"),
        ("data", "save_loc_physics"),
        ("loss", "latitude_weights"),
        ("predict", "init_cond_fast_climate"),
        ("predict", "forcing_file"),
        ("predict", "metadata"),
    ]:
        path = (conf.get(section) or {}).get(key)
        if path:
            check(f"{section}.{key}", os.path.exists(path), path)

    fixers = ["tracer_fixer", "global_mass_fixer", "global_water_fixer", "global_energy_fixer_updown"]
    outside = {f: pc.get(f, {}).get("activate_outside_model") for f in fixers if pc.get(f, {}).get("activate")}

    if args.train:
        print("\n-- training --")
        tr = conf["trainer"]
        check(
            "trainer.type is era5-gen1",
            tr.get("type") in ("era5", "era5-gen1"),
            f"{tr.get('type')!r} -- the gen2 trainer DELETES model.post_conf",
        )
        check("data.dataset_type is set", bool(conf["data"].get("dataset_type")), str(conf["data"].get("dataset_type")))
        check(
            "thread_workers == 1",
            tr.get("thread_workers") == 1,
            f"{tr.get('thread_workers')} -- ERA5_MultiStep_Batcher is stateful; >1 duplicates forecast steps",
        )
        check(
            "valid_thread_workers >= 1", (tr.get("valid_thread_workers") or 0) >= 1, str(tr.get("valid_thread_workers"))
        )
        check(
            "forecast_len uses gen1 indexing",
            conf["data"].get("forecast_len") == 2,
            f"{conf['data'].get('forecast_len')} -- era5-gen1 counts from 0, so 2 is the 3-step rollout",
        )
        check(
            "fixers run INSIDE model.forward (activate_outside_model: False)",
            all(v is False for v in outside.values()),
            str(outside),
        )
        if tr.get("load_weights"):
            ckpt = os.path.join(conf.get("save_loc", ""), "checkpoint.pt")
            check("checkpoint staged at save_loc/checkpoint.pt", os.path.exists(ckpt), ckpt)
    else:
        print("\n-- rollout --")
        check(
            "fixers run OUTSIDE the model (activate_outside_model: True)",
            all(v is True for v in outside.values()),
            str(outside),
        )

    print()
    if _failed:
        print(f"FAILED {len(_failed)} check(s): {', '.join(_failed)}")
    if _warned:
        print(f"warnings: {', '.join(_warned)}")
    if not _failed:
        print("all checks passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
