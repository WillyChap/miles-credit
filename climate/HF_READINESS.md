# CAMulator inference on HuggingFace — ease-of-use assessment

**Scenario assessed:** a user with **no NCAR/GLADE access** who downloads the
toolbox (code) and data (assets) directly from HuggingFace and wants to run
CAMulator **inference only**.

**Bottom line:** the *inference code path is genuinely self-contained* — it reads
nothing from GLADE beyond the documented `./assets/` files (verified by tracing
`Model_State.initialize_camulator`).

## Status after this branch's fixes

Most gaps below are now **addressed in code/docs**; one true blocker remains
because it requires action outside the repo.

| ID | Item | Status |
|----|------|--------|
| B1 | HuggingFace model repo not created | **OPEN** — needs maintainer to run `upload_assets.py` (added) and set the real `--repo_id`. `download_assets.py` now errors clearly on the placeholder. |
| B2 | CREDIT must be the repo's own version | **fixed** — README Install section says clone + `pip install -e ..`; `check_setup.py` reports which `credit` is imported. |
| M1 | toolbox depends on parent `credit/` | **fixed** — README Install section. |
| M2 | no non-NCAR environment | **fixed** — inference `environment.yml` (JAX/GraphCast removed) + README points at it. |
| M3 | `RunQuickClimate.sh` NCAR-shaped | **fixed** — env-var overrides (`CONDA_ENV`/`FOLD_OUT`/`AVG`), conda activation is optional, PBS header is inert under `bash`. |
| M5 | hardware/footprint undocumented | **fixed** — README states GPU + ~6.5 GB download. |
| m7 | forcing file must hold static vars | **fixed** — documented in README. |
| m8 | no preflight verify | **fixed** — `check_setup.py` (deps, CREDIT, config, assets, GPU). |
| m6 | config carries GLADE training paths | open (minor) — left absolute, clearly marked TRAINING-ONLY in the config. |

**Remaining to reach turn-key:** create the HF repo (B1). The clean-room
install+run test is now DONE (see below). Everything else a HF user needs is in
the repo.

### Clean-room new-user test (DONE — June 2026)

Simulated the full flow from a fresh `git clone` to NetCDF output: clone →
`conda env create -f environment.yml` → `pip install -e . --no-deps` → stage
assets → `check_setup.py` → 8-step rollout. **Passed end-to-end** (wrote
`pred_*.nc`) after fixing three real blockers the simulation surfaced:

1. **Missing `gcsfs`** — `credit.parser`'s import chain needs it; added to
   environment.yml.
2. **Broken torch stack** — environment.yml pinned torch 2.6.0 + torch_harmonics
   0.9.1, whose C extension is ABI-incompatible with torch 2.6 (`import
   credit.models` crashed). Repinned to the verified combo (torch 2.4.1+cu121,
   torch_harmonics 0.7.2, torch-geometric 2.6.1, numpy<2).
3. **`pip install -e .` re-breaks the env** — it re-resolves and upgrades
   numpy→2.2 / torch_harmonics→0.8. README + check_setup now use `--no-deps`.

`check_setup.py` was also hardened to import the full inference chain
(credit.models/distributed/postblock), so it now catches this ABI class of
failure (it previously passed while the rollout failed).

### End-to-end validation (run on NCAR Casper, V100)

Simulated the full user flow with assets staged into `./assets/`:
`check_setup.py` → all green; a 4-step `Quick_Climate.py` rollout → wrote
`pred_*.nc` (62 MB/step). This surfaced and fixed a real showstopper:

- **Forcing-coord grid collapse (fixed).** The cyclic forcing stores lat/lon as
  float32 while mean/std are float64; normalization inner-joined on coord values
  and collapsed latitude 192→2, crashing the rollout at step 1. `Model_State` now
  copies the normalization coords onto the forcing. *Every* user would have hit
  this — caught only by actually running, not by static review.
- Also fixed en route: `Make_Climate_Initial_Conditions.py` `torch.mps`
  AttributeError on CPU/non-Mac hosts.

---

## Original findings (for reference)

---

## What already works (verified)

- **No hidden training-data dependency.** `initialize_camulator()` opens only:
  checkpoint (`save_loc/<ckpt>`), `mean_path`/`std_path`, the IC `.pth`,
  `forcing_file`, `latitude_weights`, and `metadata` — all `./assets/`. It does
  **not** glob the ERA5/CESM training zarr (`data.save_loc`). Static model inputs
  (`z_norm`, `LANDM_COSLAT`) are read from the forcing file, not a GLADE static.
- **Single YAML driver**, all runtime paths relativized to `./assets/`.
- **Asset manifest** (README §4) lists every required file with size + purpose.
- **`download_assets.py`** pulls the manifest from a HF repo into `./assets/`.
- **Output** is plain NetCDF (6-hourly, or `--daily_mean`/`--monthly_mean`), no
  external post-processing step.
- **Bundled `credit/` parses this config correctly** — the repo's
  `credit/parser.py` guards `None` tracer thresholds (line ~596), so a
  `pip install -e .` from the repo root yields a compatible CREDIT.

---

## Blockers (a HF-only user is stuck without these)

**B1 — The HuggingFace model repo does not exist yet.**
`download_assets.py` defaults to `NCAR/camulator` and the README shows
`<user>/camulator`; neither is real. Until the repo is created and the 9 assets
(checkpoint, mean/std, statics×2, forcing, IC, era5.yaml) are uploaded with the
exact filenames in the manifest, `download_assets.py` fetches nothing.
→ *Create the HF model repo, upload assets at repo root, set the real default
`--repo_id`.*

**B2 — CREDIT must be the repo's own version, but the README doesn't say so.**
The config triggers a parser path that an *older* CREDIT build crashes on
(`float(None)` on `tracer_thres_max` — observed with `credit_feb182026`). The
repo's bundled `credit/` is fine, but a user who instead runs
`pip install miles-credit` from PyPI/main may get an incompatible parser and a
cryptic crash. The README quick-start only says `conda activate <NCAR path>` and
never says "clone this repo and `pip install -e .`".
→ *Document the install explicitly; pin/tag the CREDIT commit.*

---

## Major friction

**M1 — Toolbox is `climate/` but depends on the parent repo's `credit/` package.**
"Download the toolbox" is ambiguous: the user needs the **whole repo** (for
`credit/`), then `pip install -e .` at root, then work in `climate/`. The
`climate/README.md` never states this.
→ *Add a "0. Install" section: clone repo → `conda env create -f
../environment.yml` → `pip install -e ..` → `cd climate`.*

**M2 — No non-NCAR environment was wired into the docs.**
The quick-start hard-codes `conda activate /glade/work/.../credit-coupling-ud`.
There is now an inference `environment.yml` at the repo root (JAX/GraphCast
removed), but the README doesn't reference it.
→ *Point the README at `environment.yml`; drop the GLADE conda path as the
primary instruction.*

---

## Moderate

- **M3 — `RunQuickClimate.sh` is NCAR-shaped:** PBS headers + hard-coded
  `CONDA_ENV=/glade/...`. A cloud user must edit `CONDA_ENV` and ignore the
  `#PBS` lines. Documented only implicitly.
- **M4 — `stage_assets.sh` is NCAR-only** (symlinks GLADE). Correctly superseded
  by `download_assets.py` for HF users, but its presence can confuse.
- **M5 — Hardware/footprint undocumented:** needs a CUDA GPU (model is
  `torch.jit.trace`d) with ~6 GB free, and ~6.5 GB of downloads (4.8 GB ckpt +
  1.3 GB forcing + ~0.2 GB statics). CPU inference is untested. Add a
  "Requirements" line.

---

## Minor

- **m6** — config still carries absolute GLADE training paths (`data.save_loc`,
  etc.); documented as training-only but could read `PLACEHOLDER/...` to remove
  any doubt.
- **m7** — the forcing file must contain the static vars (`z_norm`,
  `LANDM_COSLAT`); undocumented coupling that matters only if a user swaps it.
- **m8** — no "verify your install" smoke step (e.g. `Quick_Climate.py --help`
  or a 2-step tiny run) to catch a broken env before a long rollout.

---

## Recommended path to turn-key (ordered)

1. Create `<org>/camulator` HF model repo; upload the 9 manifest files at root;
   set the real default `--repo_id` in `download_assets.py` and the README.
2. Add an **Install** section to `climate/README.md`: clone repo →
   `conda env create -f ../environment.yml` → `pip install -e ..` → `cd climate`
   → `python download_assets.py`. Pin the CREDIT commit/tag.
3. Note hardware requirements + download size; make `RunQuickClimate.sh`'s
   `CONDA_ENV`/PBS edits explicit for non-NCAR users.
4. Do one **clean-room test** on a non-NCAR GPU (or a fresh conda env with no
   GLADE on PATH) from download → run → NetCDF, and fix whatever surfaces.
