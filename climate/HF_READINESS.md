# CAMulator inference on HuggingFace — ease-of-use assessment

**Scenario assessed:** a user with **no NCAR/GLADE access** who downloads the
toolbox (code) and data (assets) directly from HuggingFace and wants to run
CAMulator **inference only**.

**Bottom line:** the *inference code path is genuinely self-contained* — it reads
nothing from GLADE beyond the documented `./assets/` files (verified by tracing
`Model_State.initialize_camulator`). But the repo is **not yet turn-key for a
HuggingFace-only user**: two hard blockers and several documentation/packaging
gaps remain. Estimated work to turn-key: ~half a day (create the HF model repo,
add clone/install instructions, one clean-room test).

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
