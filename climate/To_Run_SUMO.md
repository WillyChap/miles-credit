# Running the SUMO Supermodel

CAM6 is steered toward the consensus forecast of one or more neural-network
surrogates via the CESM nudging toolbox at 6-hourly intervals.

Three modes are supported:

| Mode | Surrogates | Coordinator | Status |
|---|---|---|---|
| `camulator` | CAMulator (CAM6 grid, PyTorch) | `sumo_coordinator.py` | Validated, 5-day run + restarts (Feb 2026) |
| `graphcast` | GraphCast (0.25° ERA5, JAX)   | `gc_coordinator.py` | New — see below |
| `both`      | CAMulator + GraphCast blended on CAM6 grid | (planned) | Server + reverse translator done; blend coordinator TODO |

---

## Architecture

```
CAM6 (CESM B-compset, Derecho)
  ├── writes h1 history every 6 h (instantaneous fincl2 fields)
  ├── pauses at sumo_barrier_after_h1 (cam_comp.F90)
  └── reads sumo_cam6_nudge.YYYY-MM-DD-SSSSS.nc → nudging toolbox

GraphCast path                      CAMulator path
─────────────                       ──────────────
gc_coordinator.py                   sumo_coordinator.py
  ├── reads h1 at T-6h, T              ├── reads h1 (SST/ICEFRAC)
  ├── sends gc_request.flag            ├── writes camulator_sst_in.nc
  └── writes dated nudge file          └── combines, writes nudge file

graphcast_server.py (GPU, JAX)      camulator_sumo_server.py (GPU, PyTorch)
  ├── builds 12h input bundle          ├── reads SST, runs inference
  ├── runs GraphCast inference         └── writes sumo_cam_state.nc
  ├── reverse-translates to CAM6
  └── writes gc_nudge_target file
```

**The CAM6 F90 barrier (`sumo_barrier_after_h1` in `cam_comp.F90`) is unchanged across modes.** Only the coordinator changes.

---

## Environment

A single conda env serves the whole stack (PyTorch + JAX + GraphCast + CREDIT):

```bash
conda activate /glade/work/wchapman/conda-envs/supermodel
```

For the older CAMulator-only path the legacy env `credit-coupling` also still works.

---

## Reference data

| Path | Purpose |
|---|---|
| `regrid/reference_data/reference_h1_prev.nc` | CAM6 h1 at T-6h, persistent. Used by `graphcast_server.py` for grid metadata + warmup. |
| `regrid/reference_data/reference_h1_t.nc` | CAM6 h1 at T, same purpose. |
| `regrid/sumo_nudge_template.nc` (or built from h1) | Structural netCDF template for nudge file writes. |
| `/glade/derecho/scratch/wchapman/jax_compilation_cache/` | XLA persistent cache. First GraphCast compile ~18 min; subsequent calls ~10 s. |

If `fincl2` changes such that the reference h1 files no longer carry every field GraphCast needs, copy a fresh pair into `regrid/reference_data/`.

---

# Mode 1 — GraphCast only (new)

Steers CAM6 toward GraphCast's 6-hour forecast at each coupling step.

## Deployment model: two independent PBS jobs

**Casper hosts the A100 GPU; Derecho hosts CESM's compute queue.**  They share /glade/derecho/scratch but otherwise know nothing about each other. So the workflow is two completely separate PBS submissions that communicate only through the run-directory file flags:

```
Casper PBS job                          Derecho PBS job
(80 GB A100)                            (CPU compute nodes)
─────────────                           ───────────────
graphcast_server.py                     CESM (case.submit)
gc_coordinator.py                          ↑↓ h1 files,
   ↑↓ nudge files                             nudge files,
         shared rundir on /glade/derecho/scratch
```

Either job can start first; the coordinator polls for `cesm_h1_ready.flag` so it just waits if CESM hasn't started yet, and CESM's nudging toolbox warns-and-continues if a nudge file is missing (only happens at step 0 by design).

### Prereqs

* A100 GPU node (cached compile: 40 GB OK; cold compile: 80 GB recommended).
* CESM case built with the existing SUMO patches (Nudge_Do=0 in `namelist_definition.xml`, `sumo_barrier_after_h1` in `cam_comp.F90`, fincl2 with all GraphCast input fields). The existing `g.e21.SUMO_CAM6_v03` case has these.
* GraphCast checkpoint: `graphcast/params/graphcast_params_GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz` (shipped).

### 1. Reset run dir (once)

```bash
cd /glade/work/wchapman/Roman_Coupling/camulator_sumo/climate
bash reset_run_dir.sh
```

Cleans both CAMulator and GraphCast lifecycle flags, restores LE2 restart symlinks.

### 2. Submit the **Casper** job (servers + coordinator)

```bash
qsub -v MODE=graphcast,RUNDIR=/glade/derecho/scratch/wchapman/g.e21.SUMO_GC_v01/run \
     /glade/work/wchapman/Roman_Coupling/camulator_sumo/climate/submit_supermodel.pbs
```

This requests `select=1:ncpus=8:ngpus=1:mem=80GB:gpu_type=a100`. The job:
1. Activates the `supermodel` env.
2. Starts `graphcast_server.py` in the background (warmup at startup; ~18 min cold, ~30 s warm cache).
3. Starts `gc_coordinator.py` in the background.
4. Enters a heartbeat loop until the coordinator exits or walltime hits.
5. On any exit (clean, signal, walltime), trap calls `run_coordinators.sh --action stop` → SIGTERM with grace, then SIGKILL.

### 3. Submit the **Derecho** job (CESM)

In a separate terminal, on any Derecho-capable login node:

```bash
cd /glade/work/wchapman/cesm/CREDIT/g.e21.SUMO_GC_v01
./case.submit
```

CESM doesn't need to wait for the Casper job — its first 6 h step writes the first `cesm_h1_ready.flag`, the coordinator picks it up, and the loop starts.

### 4. Watch progress

```bash
tail -f /glade/derecho/scratch/wchapman/g.e21.SUMO_GC_v01/run/gc_server.log
tail -f /glade/derecho/scratch/wchapman/g.e21.SUMO_GC_v01/run/gc_coordinator.log
qstat -u $USER     # both jobs visible
```

Or, from anywhere on Casper:

```bash
bash /glade/work/wchapman/Roman_Coupling/camulator_sumo/climate/run_coordinators.sh \
     --mode graphcast --rundir $RUNDIR --action status
```

### 5. Stop early

Stop the Casper-side stack:
```bash
bash climate/run_coordinators.sh --mode graphcast --rundir $RUNDIR --action stop
# or, from anywhere:
qdel <casper_jobid>      # PBS trap on the job will run the same stop
```

Stop CESM independently with `qdel <derecho_jobid>`.

### Interactive testing (single Casper node)

For development on one A100 node with no Derecho involvement, use `launch_gc_test.sh --no-case-submit` (it inherits the right PBS env and runs both server and coordinator inline; you exit with Ctrl-C). Do **not** use it for production runs — the `case.submit` call in that script assumes Derecho-side PBS access from the Casper job, which won't work.

### Step-by-step protocol per 6 h

1. CAM6 pauses at `sumo_barrier_after_h1`, writes `cesm_h1_ready.flag`.
2. `gc_coordinator` reads the h1 path; combines with the previous step's h1.
3. Coordinator writes `gc_request.flag` (with `request_id = YYYYMMDD-SSSSS`, `h1_prev=`, `h1_t=`).
4. `graphcast_server` builds the 3-timestep bundle, runs JAX inference (~10 s after warmup), reverse-translates, writes `gc_nudge_target.<id>.nc` and `gc_prediction.<id>.nc`, drops `gc_done.flag`.
5. Coordinator copies the nudge target to `sumo_cam6_nudge.1000-MM-DD-SSSSS.nc` (dated to T+6h, the time the contents are valid at).
6. Coordinator writes `coordinator_done.flag` → CAM6 unblocks → advances toward the nudge target.

**Step 0 caveat:** the very first coupling boundary has no T-6h frame to bundle with, so it is free-running. This is a one-time 6 h gap, intentional.

---

# Mode 2 — CAMulator only (existing, validated)

Unchanged from prior validated runs. See the original launch sequence below.

### 1. Reset run dir
```bash
cd /glade/work/wchapman/Roman_Coupling/camulator_sumo/climate
bash reset_run_dir.sh
```

### 2. Start CAMulator server (GPU)
```bash
conda activate /glade/work/wchapman/conda-envs/credit-coupling
cd /glade/work/wchapman/Roman_Coupling/camulator_sumo/climate

python camulator_sumo_server.py \
    --config    camulator_config.yml \
    --model_name checkpoint.pt00044.pt \
    --rundir    /glade/derecho/scratch/wchapman/g.e21.SUMO_CAM6_v03/run \
    --init_cond /glade/derecho/scratch/wchapman/CREDIT_runs/NEW_CLI_JOHN_CASPER_extended_v2/init_times/sumo_init_camulator_condition_tensor_1980-01-01T00Z.pth \
    --sumo --sumo_vars U V --sumo_tau 6.0 \
    --save_atm_nc camulator_out --daily_mean
```

Wait for: `Server ready — waiting for CESM go.flag ...`

### 3. Start coordinator (login or CPU node)
```bash
python sumo_coordinator.py \
    --rundir    /glade/derecho/scratch/wchapman/g.e21.SUMO_CAM6_v03/run \
    --cam6dir   /glade/derecho/scratch/wchapman/g.e21.SUMO_CAM6_v03/run \
    --cam6_case g.e21.SUMO_CAM6_v03 \
    --vars U V --alpha_cam 0.5 \
    --start_ymd 19800101
```

### 4. Submit CESM
```bash
cd /glade/work/wchapman/cesm/CREDIT/g.e21.SUMO_CAM6_v03/
./case.submit
```

---

# Mode 3 — Supermodel (CAMulator + GraphCast blended) — TODO

Single coordinator (`supermodel_coordinator.py`) drives both servers in parallel and blends predictions via `regrid.blender.blend()`. Blender already supports single-source pass-through, so once the coordinator is written, all three modes share the same downstream nudge-write path.

Default blend: 50/50 each variable, every level. Alternative `stratosphere_lean_blend_config()` ramps to 80-90 % CAMulator above 50 hPa where GraphCast's training resolution drops.

---

## Common: setup, reset, prerequisites

### CESM sandbox
```
/glade/work/wchapman/JE_help_cnn/camulator_sandbox_fluxupdown/
```
Three-part `cime_comp_mod.F90` SST=0 fix (MEMORY.md), Nudge_Do XML registration, `sumo_barrier_after_h1` in `cam_comp.F90`. **No further rebuild needed for namelist-only changes.**

### Conda env (unified)
```bash
conda activate /glade/work/wchapman/conda-envs/supermodel    # JAX + PyTorch
# legacy CAMulator-only env still works:
conda activate /glade/work/wchapman/conda-envs/credit-coupling
```

### CESM case (CAM6 side, shared across modes)
```bash
cd /glade/work/wchapman/Roman_Coupling/camulator_sumo/climate
bash setup_SUMO_B_case.sh   # first time only; lands at g.e21.SUMO_CAM6_v03
```

The case's `user_nl_cam` already has:
- `Nudge_Model=.true.`, `Nudge_File_Template='sumo_cam6_nudge.%y-%m-%d-%s.nc'`
- `Nudge_Do = 0` (standard Newtonian, **NOT** the ML-tendency hack)
- `Nudge_Force_Opt = 1`, `Nudge_Ucoef = Nudge_Vcoef = 0.3`
- `fincl2` with every GraphCast input field

### Reset between attempts
```bash
bash reset_run_dir.sh
```
Removes case output, re-symlinks LE2 restart data, copies fresh rpointers, **cleans both CAMulator and GraphCast lifecycle flags**.

---

## Key files

| File | Location | Purpose |
|---|---|---|
| `setup_SUMO_B_case.sh` | `climate/` | Create + configure CESM case (CAM6 side) |
| `reset_run_dir.sh` | `climate/` | Reset run dir between attempts (covers both modes) |
| `camulator_sumo_server.py` | `climate/` | CAMulator GPU server (PyTorch) |
| `sumo_coordinator.py` | `climate/` | CAMulator coordinator (existing) |
| `gc_coordinator.py` | `climate/` | GraphCast coordinator (new) |
| `launch_gc_test.sh` | `climate/` | Orchestrates GC server + coord + CESM, kill-on-exit |
| `graphcast_server.py` | `regrid/` | Long-lived GC inference server (JAX) |
| `run_graphcast_inference.py` | `regrid/` | One-shot GC inference (uses JAX cache) |
| `gc_to_nudge.py` | `regrid/` | Standalone: GC prediction → CAM6 nudge file |
| `inspect_gc_prediction.py` | `regrid/` | Inspect GC output, optional ERA5 compare |
| `reference_data/reference_h1_{prev,t}.nc` | `regrid/` | Persistent h1 pair for stencil + warmup |
| CESM case | `/glade/work/wchapman/cesm/CREDIT/g.e21.SUMO_CAM6_v03/` | CAM6 case |
| Run directory | `/glade/derecho/scratch/wchapman/g.e21.SUMO_CAM6_v03/run/` | Shared flag/file exchange |

---

## Nudge file convention

CAM6's `nudging.F90` hard-codes `Nudge_Climo_Year = 1000`. All nudge files write year 1000: `sumo_cam6_nudge.1000-MM-DD-SSSSS.nc`. The MM/DD/SSSSS come from the actual simulation time, T+6h-labeled (the time the contents are valid at). With the F90 barrier in place, file label and contents match exactly — no T+12h workaround needed.

---

## JAX compilation cache

Set automatically by `graphcast_server.py` and `run_graphcast_inference.py`:

```
JAX_COMPILATION_CACHE_DIR=/glade/derecho/scratch/wchapman/jax_compilation_cache
```

**Cold compile populates the cache (~18 min) → every subsequent call is ~10 s.** The cache is keyed on `(input shapes, JAX version, jaxlib version, GPU model)`; change any of those → recompile. Scratch may be purged after weeks of no access; if the cache disappears, the first call pays the compile tax again.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `gc_server_ready.flag` never appears | First-time cold XLA compile (~15-20 min) | Wait. Check `gc_server.log` for `[gc-server] warmup complete` |
| GC server OOM at compile | V100 32 GB too small for 0.25°/37-lvl | Use A100 (40 GB minimum; 80 GB ample) |
| Memory blowup at gigabyte scale during reverse translate | Old vertical_interp.py bug (`field_plev[k0]`) | Fixed with `np.take_along_axis` (Jun 2026). Run `python -m regrid.tests.test_translator_blocks` to confirm |
| `_FillValue = NaN` issues in CAM6 reading nudge | xarray default fill on coords | Fixed in `reverse_translate.write_nudge_file` — encoding suppresses fills |
| Coordinator hangs at start | GC server not yet READY | Coordinator waits up to `--gc_ready_timeout` (30 min default). Check `gc_server.log` |
| CAMulator server stuck on `Waiting for go.flag` | Normal — coordinator writes after first h1 | Wait for CAM6's first 6 h step to complete |
| `forrtl severe 47: write to READONLY file` | rpointer.* symlinked instead of copied | Run `reset_run_dir.sh` (uses `cp`) |
| Nudge files not found by CAM6 | Wrong year in filename | Coordinators hardcode `Nudge_Climo_Year = 1000` |
| `SST=0` in coupled run | `xao_ax` null pointer in `cime_comp_mod.F90` | Three-part fix in CESM sandbox; CAM6 case already built against it |
| Stale `gc_server.pid` after SIGKILL | Expected — graceful exit cleans, ungraceful does not | `kill -9 $(cat gc_server.pid)` or `rm` the file; next launcher run resets |

---

## Smoke test (server only, no CESM)

Validate the GraphCast server in isolation before integration:

```bash
conda activate /glade/work/wchapman/conda-envs/supermodel
cd /glade/work/wchapman/Roman_Coupling/camulator_sumo
python -m regrid.smoke_test_graphcast_server
```

Three phases: full lifecycle (ready + 2 requests + stop), SIGTERM mid-idle, SIGKILL (rude — must not corrupt rundir). First run ~20 min (cold JIT); subsequent runs ~3 min. Populates the JAX cache as a side effect.
