"""
profile_camulator.py
--------------------
Profile Camulator forward pass — pad=0 vs pad=48, small vs big model.

Usage:
    python profile_camulator.py [--config PATH] [--batch_size N]
"""

import argparse
import time

import torch
import yaml
from torch.profiler import ProfilerActivity, profile, record_function


def load_camulator(conf_override, device):
    from credit.models import load_model
    mc = conf_override.copy()
    mc["post_conf"] = {"activate": False}
    model = load_model({"model": mc})
    return model.to(device).eval()


def make_input(mc, batch_size, device):
    total_ch = mc["channels"] * mc["levels"] + mc["surface_channels"] + mc["input_only_channels"]
    return torch.randn(batch_size, total_ch, mc["image_height"], mc["image_width"],
                       device=device, dtype=torch.float32)


def time_forward(model, x, n_warmup=5, n_runs=10):
    """Return mean forward-pass wall time in milliseconds."""
    with torch.no_grad(), torch.autocast("cuda"):
        for _ in range(n_warmup):
            _ = model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000


def profile_forward(model, x, label):
    """Run torch profiler and print top CUDA ops."""
    with torch.no_grad(), torch.autocast("cuda"):
        for _ in range(3):
            _ = model(x)
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
        record_shapes=False,
    ) as prof:
        with torch.no_grad(), torch.autocast("cuda"):
            for _ in range(5):
                with record_function("forward"):
                    _ = model(x)
                torch.cuda.synchronize()

    print(f"\n{'='*72}")
    print(f"  {label}  —  Top 20 ops by CUDA self-time")
    print(f"{'='*72}")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    trace_path = f"profile_results/trace_{label}.json"
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace → {trace_path}  (open at ui.perfetto.dev)")
    return prof


def run(conf, mc_base, batch_size, device, label, pad_lat, pad_lon, compile_model=False):
    mc = mc_base.copy()
    mc["padding_conf"] = mc_base["padding_conf"].copy()
    mc["padding_conf"]["activate"] = (pad_lat > 0 or pad_lon > 0)
    mc["padding_conf"]["pad_lat"] = pad_lat
    mc["padding_conf"]["pad_lon"] = pad_lon

    print(f"\n{'─'*72}")
    print(f"  Building: {label}")
    print(f"  dim={mc['dim']}  depth={mc['depth']}")
    print(f"  pad_lat={pad_lat}  pad_lon={pad_lon}  →  padded {mc['image_height']+2*pad_lat}×{mc['image_width']+2*pad_lon}")
    print(f"  compiled={compile_model}")

    model = load_camulator(mc, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {n_params:.1f}M")

    if compile_model:
        print("  Compiling with torch.compile (this takes ~60s first time)...")
        model = torch.compile(model)

    x = make_input(mc, batch_size, device)

    ms = time_forward(model, x)
    print(f"  Forward pass: {ms:.1f} ms  ({batch_size} samples)")

    profile_forward(model, x, label)

    del model
    torch.cuda.empty_cache()
    return ms


def main():
    import os
    os.makedirs("profile_results", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/glade/derecho/scratch/wchapman/b_credit_runs/camulator_config_casper.yml")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  ({torch.cuda.get_device_name(0)})")
    print(f"Batch size: {args.batch_size}")

    mc_base = conf["model"].copy()

    results = {}

    # Valid pads: (192 + 2*pad) must be divisible by 48 (4x stride-2 × local_window=3)
    # pad=0  → 192  ✓   pad=24 → 240 ✓   pad=48 → 288 ✓

    SMALL_DIM   = [128, 256, 512, 1024]
    SMALL_DEPTH = [2, 2, 12, 2]

    # --- baseline: small model, pad=48, original kernels [4,8,16,32] ---
    mc = mc_base.copy()
    mc["dim"]   = SMALL_DIM
    mc["depth"] = SMALL_DEPTH
    ms = run(conf, mc, args.batch_size, device, "small_pad48_k32", pad_lat=48, pad_lon=48)
    results["small_pad48_k32"] = ms

    # --- small model, pad=24, kernels [4,8,16,24] (largest kernel = pad size) ---
    mc = mc_base.copy()
    mc["dim"]   = SMALL_DIM
    mc["depth"] = SMALL_DEPTH
    mc["cross_embed_kernel_sizes"] = [[4, 8, 16, 24], [2, 4], [2, 4], [2, 4]]
    ms = run(conf, mc, args.batch_size, device, "small_pad24_k24", pad_lat=24, pad_lon=24)
    results["small_pad24_k24"] = ms

    # --- small model, pad=24, kernels [4,8,16] (drop 24/32 entirely) ---
    mc = mc_base.copy()
    mc["dim"]   = SMALL_DIM
    mc["depth"] = SMALL_DEPTH
    mc["cross_embed_kernel_sizes"] = [[4, 8, 16], [2, 4], [2, 4], [2, 4]]
    ms = run(conf, mc, args.batch_size, device, "small_pad24_k16", pad_lat=24, pad_lon=24)
    results["small_pad24_k16"] = ms

    # --- small model, pad=0, kernels [4,8] (no boundary padding at all) ---
    mc = mc_base.copy()
    mc["dim"]   = SMALL_DIM
    mc["depth"] = SMALL_DEPTH
    mc["cross_embed_kernel_sizes"] = [[4, 8], [2, 4], [2, 4], [2, 4]]
    ms = run(conf, mc, args.batch_size, device, "small_pad0_k8", pad_lat=0, pad_lon=0)
    results["small_pad0_k8"] = ms

    # --- torch.compile variants (best uncompiled config repeated with compile) ---
    print("\n\n=== torch.compile tests (first pass triggers JIT, ~60s) ===")
    for label, pad, kernels in [
        ("small_pad48_k32_compiled",  48, [4, 8, 16, 32]),
        ("small_pad24_k24_compiled",  24, [4, 8, 16, 24]),
        ("small_pad24_k16_compiled",  24, [4, 8, 16]),
        ("small_pad0_k8_compiled",     0, [4, 8]),
    ]:
        mc = mc_base.copy()
        mc["dim"]   = SMALL_DIM
        mc["depth"] = SMALL_DEPTH
        mc["cross_embed_kernel_sizes"] = [kernels, [2, 4], [2, 4], [2, 4]]
        ms = run(conf, mc, args.batch_size, device, label,
                 pad_lat=pad, pad_lon=pad, compile_model=True)
        results[label] = ms

    print(f"\n{'='*65}")
    print("  SUMMARY  (batch_size={})".format(args.batch_size))
    print(f"{'='*65}")
    baseline = results["small_pad48_k32"]
    for label, ms in results.items():
        speedup = f"  {baseline/ms:.2f}×" if label != "small_pad48_k32" else "  baseline"
        compiled = " [compiled]" if "compiled" in label else ""
        print(f"  {label:<36}  {ms:7.1f} ms{speedup}{compiled}")
    print(f"\n  compile() gain per config:")
    for base_label in ["small_pad48_k32", "small_pad24_k24", "small_pad24_k16", "small_pad0_k8"]:
        comp_label = base_label + "_compiled"
        if base_label in results and comp_label in results:
            gain = (results[base_label] - results[comp_label]) / results[base_label] * 100
            print(f"    {base_label:<28}  {gain:+.0f}%")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
