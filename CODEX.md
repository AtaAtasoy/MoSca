# CODEX Context

## What This Repository Is

MoSca is a monocular 4D reconstruction system with a two-stage operator workflow:

1. `mosca_precompute.py` builds 2D priors from a workspace of frames.
2. `mosca_reconstruct.py` solves camera/static geometry first, then dynamic scaffold + photometric reconstruction.

There is also a lighter camera-only path:

1. `mosca_precompute.py --skip_dynamic_resample`
2. `lite_moca_reconstruct.py`


## Default Workflow

Use [example.sh](example.sh) as the canonical run pattern.

Minimal demo flow:

```bash
CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/duck
CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/duck
```

Important variants from `example.sh`:

- `duck`: uses demo profile defaults.
- `shiba`: overrides `--tap_mode=bootstapir --boundary_enhance_th=-1.0`.
- `breakdance-flare` and `train`: override `--dep_mode=uni --tap_mode=bootstapir --boundary_enhance_th=-1.0`.

Interpretation:

- Demo prep defaults are `dep_mode=depthcrafter`, `tap_mode=spatracker`, `flow_mode=raft`.
- `boundary_enhance_th > 0` creates `*_depth_sharp` and reconstruction may auto-pick that directory.
- CLI overrides are merged on top of YAML via OmegaConf dotlist parsing.

## Entry Points

- [mosca_precompute.py](mosca_precompute.py): loads frames from `WS/images` or directly from an `.mp4`, runs depth/TAP/flow/EPI preprocessing, and optionally resamples more TAP tracks in dynamic regions.
- [lite_moca_reconstruct.py](lite_moca_reconstruct.py): camera-focused reconstruction using MoCa bundle adjustment only.
- [mosca_reconstruct.py](mosca_reconstruct.py): full reconstruction pipeline.
- [recon_utils.py](recon_utils.py): workspace/log setup, auto-discovery of depth/TAP artifacts, track identification helpers.
- [render_experiment.py](render_experiment.py): post-training renderer for train/test camera trajectories with `full`, `static`, or `dynamic` output and PNG/MP4/GIF export.
- [example.sh](example.sh): fastest way to understand intended operator usage.
- [readme.md](readme.md): install notes, dataset modes, and paper-level context.

## Full Pipeline Structure

### 1. Precompute

`mosca_precompute.py`:

- creates or reuses `WS/images`
- saves `input.mp4`
- writes `preprocess.log` and `precompute_commandline_args.txt`
- runs foundational models through `lib_prior.moca_processor.MoCaPrep`
- generates depth, TAP tracks, optical flow, and EPI signals
- optionally resamples denser dynamic TAP tracks using EPI-derived motion regions

Main implementation:

- [lib_prior/moca_processor.py](lib_prior/moca_processor.py)
- [lib_prior/prior_loading.py](lib_prior/prior_loading.py)

Supported prior backends visible in code:

- Depth: `metric3d`, `uni`, `depthcrafter`
- TAP: `spatracker`, `cotracker`, `bootstapir`
- Flow: `raft`

External-prior path now supported for VIPE-style inputs:

- `mosca_precompute.py` can import external depth priors instead of running a depth model when `known_depth_path` and `known_depth_format` are set.
- `mosca_precompute.py` can also use known intrinsics during preprocessing when `known_camera_intrinsics_path` is provided.
- `lite_moca_reconstruct.py` supports `known_camera_mode: init` with `known_camera_format: vipe`, initializing MoCa from external camera poses and intrinsics.
- VIPE camera priors can come from legacy `.npz` files or scene-style `pred_traj.txt` / `pred_intrinsics.txt` exports.
- The helper implementation lives in [data_utils/known_camera_helpers.py](data_utils/known_camera_helpers.py).

### 2. Static Reconstruction

`lite_moca_reconstruct.static_reconstruct()`:

- loads RGB, depth, EPI, TAP, VOS-style masks through `Saved2D`
- auto-detects depth directory and TAP mode unless explicitly pinned in config
- runs `lib_moca.moca.moca_solve(...)`
- writes bundle-adjusted cameras and bundle state under `logs/.../bundle`

MoCa code lives mainly in:

- [lib_moca/moca.py](lib_moca/moca.py)
- [lib_moca/camera.py](lib_moca/camera.py)
- [lib_moca/bundle.py](lib_moca/bundle.py)

### 3. Optional Static Photometric Warmup

`mosca_reconstruct.photometric_warmup()`:

- reconstructs static background GS first if `photo_static_warm_steps >= 0`
- saves `photo_warmup_rendered.npz`
- may overwrite `bundle/bundle_cams.pth` after warmup and keep previous BA cameras as `bundle_cams_ba.pth`

### 4. Dynamic Scaffold Reconstruction

`mosca_reconstruct.scaffold_reconstruct()`:

- reloads bundle-scaled depth and TAP tracks
- re-identifies static/dynamic tracks from EPI and optionally photometric error
- extracts dynamic curves
- initializes and optimizes a `MoSca` scaffold
- saves `track_identification.npz` and `mosca/mosca.pth`

Core files:

- [lib_mosca/mosca.py](lib_mosca/mosca.py)
- [lib_mosca/dynamic_solver.py](lib_mosca/dynamic_solver.py)
- [lib_mosca/dynamic_solver_utils.py](lib_mosca/dynamic_solver_utils.py)

### 5. Photometric Reconstruction

`mosca_reconstruct.photometric_reconstruct()`:

- loads cameras, scaffold, static/dynamic track IDs, flow, and priors
- builds static and dynamic Gaussian models
- runs joint photometric optimization
- emits the final reconstruction assets and evaluation outputs

Core files:

- [lib_mosca/photo_recon.py](lib_mosca/photo_recon.py)
- [lib_mosca/static_gs.py](lib_mosca/static_gs.py)
- [lib_mosca/dynamic_gs.py](lib_mosca/dynamic_gs.py)
- [lib_mosca/photo_recon_utils.py](lib_mosca/photo_recon_utils.py)

## Workspace Contract

For precompute, `--ws` may be:

- a directory containing `images/`
- an `.mp4` file

For reconstruction, `--ws` should be a workspace directory containing `images/` and precomputed priors.

Common workspace contents after precompute:

- `images/`
- `input.mp4`
- `preprocess.log`
- `precompute_commandline_args.txt`
- one or more depth directories such as `depthcrafter_depth`, `depthcrafter_depth_sharp`, `uni_depth`, `metric3d_depth`
- TAP files like `*uniform*_<tap_mode>_tap.npz` and possibly dynamic resample TAP files
- flow/EPI/VOS artifacts used by `Saved2D`

When VIPE depth import is used, expect artifacts like:

- `vipe_depth/`
- `vipe_depth.mp4`
- `uniform_dep=vipe_<tap_mode>_tap.npz`
- `dynamic_dep=vipe_<tap_mode>_tap.npz`

Common workspace contents after reconstruct:

- `logs/<exp_name>_<gs_backend>_<timestamp>/`
- `bundle/`
- `mosca/`
- `src_backup/`
- visualizations such as GIFs and MP4s
- evaluation text files

`setup_recon_ws()` copies source snapshots into `src_backup`, so runs are designed to be somewhat self-contained.

## Auto-Discovery Rules Worth Remembering

`recon_utils.auto_get_depth_dir_tap_mode()` matters a lot.

Depth selection:

- If `fit_cfg.depth_dirname` is unset, reconstruction scans `WS/*_depth`.
- If multiple depth directories exist, it tries to disambiguate with this priority: `gt`, `sensor`, `sharp`, `depthcrafter`.
- This means `boundary_enhance_th > 0` can change which depth directory gets picked later.

TAP selection:

- If `fit_cfg.tap_mode` is unset, reconstruction scans `WS/*uniform*tap.npz`.
- It expects exactly one matching mode.

Practical consequence:

- When experimenting with multiple priors in the same workspace, explicitly set `tap_mode` and/or `depth_dirname` in the fit config or CLI to avoid ambiguous auto-picks.

## Config Layout

Profiles live under [profile](profile):

- `profile/demo`: sample wild-scene settings used by `example.sh`
- `profile/iphone`
- `profile/nvidia`
- `profile/sintel`
- `profile/tum`
- `profile/vipe`: external-prior examples for VIPE camera/depth inputs

High-signal configs for first read:

- [profile/demo/demo_prep.yaml](profile/demo/demo_prep.yaml)
- [profile/demo/demo_fit.yaml](profile/demo/demo_fit.yaml)

Config merge behavior:

- file config is loaded with `OmegaConf.load(...)`
- unknown CLI args are converted from `--key=value` to OmegaConf dotlist
- CLI wins over YAML

Examples:

```bash
python mosca_precompute.py --cfg profile/demo/demo_prep.yaml --ws demo/shiba --tap_mode=bootstapir --boundary_enhance_th=-1.0
python mosca_reconstruct.py --cfg profile/demo/demo_fit.yaml --ws demo/shiba --tap_mode=bootstapir
```

VIPE example profiles added during this session:

```bash
python mosca_precompute.py --cfg profile/vipe/cowboy_cat_prep_vipe_depth.yaml --ws /tmp/mosca_vipe_cowboy_cat
python lite_moca_reconstruct.py --cfg profile/vipe/cowboy_cat_fit_init.yaml --ws /tmp/mosca_vipe_cowboy_cat
python mosca_reconstruct.py --cfg profile/vipe/cowboy_cat_fit_init.yaml --ws /tmp/mosca_vipe_cowboy_cat
```

Practical VIPE run example using the checked-in cowboy-cat profiles:

```bash
CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg profile/vipe/cowboy_cat_prep_vipe_depth.yaml --ws ./runs/vipe_cowboy_cat
CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg profile/vipe/cowboy_cat_fit_init.yaml --ws ./runs/vipe_cowboy_cat
```

The `cowboy_cat_fit_init.yaml` profile initializes MoCa from external VIPE camera poses and intrinsics via:

- `known_camera_mode: init`
- `known_camera_format: vipe`
- `known_camera_pose_path`
- `known_camera_intrinsics_path`

The checked-in cowboy-cat VIPE profiles use the scene-style export bundle under
`/dss/dssmcmlfs01/pn25pi/pn25pi-dss-0000/di35dov/monst3r/demo_tmp/vipe_results/121frames/cowboy-cat/scene/normalized_nofilter`:

- depth maps: `frame_####.npy`
- camera trajectory: `pred_traj.txt`
- intrinsics: `pred_intrinsics.txt`

## Visualization Commands

Render the final reconstruction from training cameras:

```bash
python render_experiment.py \
  --logdir ./runs/vipe_cowboy_cat/logs/cowboy_cat_vipe_init_native_add3_<timestamp> \
  --ws ./runs/vipe_cowboy_cat \
  --cfg ./profile/vipe/cowboy_cat_fit_init.yaml \
  --camera_set train \
  --region full
```

Render only the static or dynamic component:

```bash
python render_experiment.py --logdir <run_logdir> --ws <workspace> --cfg <fit_cfg> --camera_set train --region static
python render_experiment.py --logdir <run_logdir> --ws <workspace> --cfg <fit_cfg> --camera_set train --region dynamic
```

Render explicit test-camera `.npz` files:

```bash
python render_experiment.py \
  --logdir <run_logdir> \
  --cfg <fit_cfg> \
  --camera_set test \
  --region full \
  --test_camera_pose_path /path/to/test_pose.npz \
  --test_camera_intrinsics_path /path/to/test_intrinsics.npz
```

Visualize input, optimized, and optional test cameras with Dreifus + PyVista:

```bash
python visualize_mosca_cameras.py \
  --logdir <run_logdir> \
  --cfg <fit_cfg> \
  --ws <workspace> \
  --show_input_cameras \
  --show_optimized_cameras \
  --show_test_cameras \
  --show_static_points
```

Visualize explicit test-camera `.npz` trajectories in the same scene:

```bash
python visualize_mosca_cameras.py \
  --logdir <run_logdir> \
  --cfg <fit_cfg> \
  --ws <workspace> \
  --show_optimized_cameras \
  --show_test_cameras \
  --test_camera_pose_path /path/to/test_pose.npz \
  --test_camera_intrinsics_path /path/to/test_intrinsics.npz
```

## Modules By Responsibility

- [lib_prior](lib_prior): foundational-model wrappers and preprocessing.
- [lib_moca](lib_moca): camera/depth alignment and static bundle adjustment.
- [lib_mosca](lib_mosca): dynamic scaffold + Gaussian reconstruction logic.
- [lib_render](lib_render): Gaussian rendering backends and CUDA extensions.
- [data_utils](data_utils): dataset-specific helpers.
- [eval_utils](eval_utils): evaluation data and helpers.

## One Level Deeper: Code Architecture

### `lib_prior`: how priors are created and reloaded

- [lib_prior/moca_processor.py](lib_prior/moca_processor.py) is the main preprocessing operator.
- `MoCaPrep.__init__()` eagerly loads the selected depth, TAP, and flow wrappers and hardcodes device use around `cuda:0`, with the expectation that GPU selection is controlled externally via `CUDA_VISIBLE_DEVICES`.
- `MoCaPrep.process()` is the canonical precompute sequence:
  1. create/reuse workspace and persist inputs
  2. compute or import depth
  3. optionally compute RAFT flow and EPI
  4. compute a uniform TAP set named like `uniform_dep=<dep_mode>_*_tap.npz`
- `MoCaPrep.compute_tap()` handles backend-specific TAP behavior.
  - `spatracker` uses depth and boundary masking and can consume known intrinsics.
  - `cotracker` and `bootstapir` follow the same save contract but do not use the same depth-aware branch.
- `boundary_enhance_th > 0` causes sharpened depth output and TAP generation may switch to `*_depth_sharp` automatically.
- External depth priors are wired through `external_depth_src` and `external_depth_format` and flow through the same workspace artifact contract as model-generated priors.

- [lib_prior/prior_loading.py](lib_prior/prior_loading.py) defines `Saved2D`, which is the common in-memory representation used by reconstruction stages.
- `Saved2D` always loads RGB first, then layers additional workspace artifacts on top.
- Key loader methods:
  - `load_dep()`: loads depth and creates a Laplacian-based boundary validity mask.
  - `load_track()`: finds matching TAP `.npz` files, concatenates them, rounds image coordinates to integer pixels, filters with depth validity, and stores `track` plus `track_mask`.
  - `load_flow()`: loads all saved pairwise flow files and indexes them by `(src_t, dst_t)`.
  - `load_epi()` and `load_vos()`: attach optional EPI and VOS data when present.
- `normalize_depth()` rescales depth so the valid-depth median becomes `dep_median` and also rescales 3D SpaTracker tracks if present.
- `rescale_perframe_depth_from_bundle()` later multiplies depth by the per-frame bundle scale from `bundle/bundle.pth`.
- `register_2d_identification()` stores per-pixel static/dynamic masks.
- `register_track_indentification()` stores per-track static/dynamic labels used by scaffold and photometric stages.

### `lib_moca`: camera model and static bundle adjustment

- [lib_moca/camera.py](lib_moca/camera.py) defines `MonocularCameras`, the project-wide camera object.
- It stores intrinsics as relative focal parameters plus principal-point ratios, not raw pixel intrinsics.
- Extrinsics can be represented in two modes:
  - delta mode: frame-to-frame transforms accumulated by `forward_T()`
  - independent mode: one `T_wc` per frame
- Bundle adjustment commonly starts in delta mode and then switches to independent mode with `disable_delta()`.
- The class also owns projection, backprojection, transform, and smoothness-loss utilities, so most geometry code depends on it directly.

- [lib_moca/moca.py](lib_moca/moca.py) is the static reconstruction coordinator.
- `moca_solve()` does the following:
  1. classify tracks as static/dynamic from precomputed EPI or track-derived epipolar analysis
  2. select static tracks and query their depth
  3. estimate focal length if no camera prior is provided
  4. initialize `MonocularCameras`
  5. run static BA through `compute_static_ba()`
- If no RAFT EPI is available, it saves `tracker_epi.npz` so later stages can reuse track-based epipolar analysis.
- When GT or external cameras are used, the solve path can skip the focal-search initialization but still uses the same BA stage afterward.

- [lib_moca/bundle.py](lib_moca/bundle.py) contains the actual static BA loop.
- `compute_static_ba()` jointly optimizes:
  - camera pose
  - camera focal
  - per-frame depth scale
  - optional per-track depth correction
- The core losses are reprojection/flow consistency plus depth consistency, with optional camera smoothness regularization.
- Robustification is built from depth magnitude and 3D point-trajectory variance, so BA behavior depends strongly on the initial normalized depth distribution.
- On save, BA writes:
  - `bundle/bundle_cams.pth`: camera checkpoint
  - `bundle/bundle.pth`: per-frame `dep_scale`, optional `dep_correction`, and the static track subset used by BA
- Later stages depend on `dep_scale` being replayed via `Saved2D.rescale_perframe_depth_from_bundle()`.

### `lib_mosca`: dynamic curves, scaffold, and photometric solve

- [lib_mosca/dynamic_solver.py](lib_mosca/dynamic_solver.py) converts dynamic 2D/3D tracks into scaffold-ready 3D curves.
- `get_dynamic_curves()` is the critical entry point.
  - For 2D tracks, it backprojects through bundle-scaled depth and fills missing time slots with line-segment interpolation.
  - For SpaTracker 3D tracks, it can blend tracker-provided 3D with depth-unprojected geometry, then refilter outliers.
- It has several important cleanup passes:
  - Open3D outlier filtering
  - shaking detection
  - SpaTracker consistency filtering
  - optional full-curve removal when too unstable
- This stage is where many bad dynamic tracks are culled before the scaffold is built.

- [lib_mosca/mosca.py](lib_mosca/mosca.py) defines the dynamic scaffold object `MoSca`.
- `MoSca` stores time-varying nodes, visibility/certainty masks, topology, per-node sigma, and skinning configuration.
- Initialization computes:
  - spatial unit, either auto-estimated from curve spacing or hard-set
  - topology graph from node proximity and visibility
  - initial node rotations from node trajectories
- The scaffold supports multi-level ARAP regularization and either dual-quaternion blending (`dqb`) or LBS, with `dqb` as the normal path.
- The saved `mosca/mosca.pth` checkpoint is later reloaded and reconfigured for photometric optimization.

- [lib_mosca/photo_recon.py](lib_mosca/photo_recon.py) is the joint rendering and optimization driver.
- `DynReconstructionSolver` owns:
  - logging/viz directories
  - 2D FG/BG mask construction from nearest dynamic curves
  - normal estimation from depth when required by the backend
  - static and dynamic Gaussian initialization
  - the main photometric training loop
- `identify_fg_mask_by_nearest_curve()` derives per-pixel static/dynamic masks by nearest scaffold curve in world space, then writes those masks back into `Saved2D`.
- `get_static_model()` and `get_dynamic_model()` materialize Gaussian primitives from depth, RGB, and scaffold state.
- `photometric_fit()` is the final long optimization loop and saves:
  - `<phase>_s_model_<backend>.pth`
  - `<phase>_d_model_<backend>.pth` when dynamic GS is active
  - `<phase>_cam.pth`
- In the full pipeline, the final phase name is effectively `photometric`, which is why evaluation later reads `photometric_cam.pth`.

## Data And Artifact Flow

### Workspace to `Saved2D`

- Precompute writes files to the workspace in formats that `Saved2D` discovers by convention rather than explicit manifests.
- Reconstruction then rebuilds its working state by chaining:
  - `load_epi()`
  - `load_dep(...)`
  - `normalize_depth(...)`
  - `recompute_dep_mask(...)`
  - `load_track(...)`
  - `load_vos()` and `load_flow()` as needed

### Depth scaling lifecycle

- Raw depth enters from a model or external prior during precompute.
- `Saved2D.normalize_depth()` rescales global depth so the valid median becomes `dep_median`, usually `1.0`.
- Static BA learns a per-frame `dep_scale` and stores it in `bundle/bundle.pth`.
- Dynamic and photometric stages replay that scale with `rescale_perframe_depth_from_bundle()`.
- Practical consequence: if depth values look wrong in later stages, check both the initial normalization and the saved bundle scale, not only the raw depth files.

### Track lifecycle

- Precompute always creates a uniform TAP set and may also create a dynamic-resampled TAP set.
- Reconstruction static BA intentionally loads only the uniform TAP files using a pattern like `*uniform*{tap_mode}`.
- Later stages also use the uniform set for static/dynamic identification, then save the classification in `track_identification.npz`.
- Dynamic resample TAP files exist for denser foreground coverage during preprocessing, but the main reconstruction path still relies heavily on the uniform set unless code is changed explicitly.

### Log directory contents that matter downstream

- `setup_recon_ws()` creates `WS/logs/<exp>_<backend>_<timestamp>/`.
- It snapshots source into `src_backup/` and writes `fit_commandline_args.txt`.
- The downstream-critical files are:
  - `bundle/bundle_cams.pth`
  - `bundle/bundle.pth`
  - `photo_warmup_rendered.npz` if warmup is enabled
  - `track_identification.npz`
  - `mosca/mosca.pth`
  - `photometric_cam.pth`
  - final static/dynamic Gaussian checkpoints

## Operational Gotchas

- GPU assumption: `MoCaPrep` internally binds model work to `cuda:0`, so the intended way to choose a GPU is `CUDA_VISIBLE_DEVICES`, not code edits.
- Workspace ambiguity: the loader contract is glob-based. Extra `*_depth` directories or multiple `*uniform*tap.npz` variants in one workspace can silently change later behavior or trigger assertions.
- Integerized TAP coordinates: `Saved2D.load_track()` rounds image coordinates to integer pixels early. Any future code that assumes subpixel TAP positions should verify this behavior first.
- Depth masks are boundary-sensitive: both preprocessing and reconstruction re-run Laplacian boundary masking, so changes to `depth_boundary_th` can ripple through track filtering, BA, scaffold extraction, and GS initialization.
- Warmup mutates camera outputs: if photometric warmup is enabled, `bundle/bundle_cams.pth` is overwritten and the original BA cameras move to `bundle_cams_ba.pth`.
- Final eval path: full reconstruction evaluation looks for `photometric_cam.pth`, not the bundle camera file.

If time is limited, read in this order:

1. [example.sh](example.sh)
2. [profile/demo/demo_prep.yaml](profile/demo/demo_prep.yaml)
3. [profile/demo/demo_fit.yaml](profile/demo/demo_fit.yaml)
4. [mosca_precompute.py](mosca_precompute.py)
5. [lite_moca_reconstruct.py](lite_moca_reconstruct.py)
6. [mosca_reconstruct.py](mosca_reconstruct.py)
7. [recon_utils.py](recon_utils.py)

## Environment Assumptions

- Python 3.10 via conda in [install.sh](install.sh)
- PyTorch 2.1.0 + CUDA 11.8 in the provided install path
- Linux/Ubuntu-oriented setup
- heavy GPU usage
- several third-party model weights expected under `weights/`

Rendering backend:

- [lib_render/render_helper.py](lib_render/render_helper.py) reads `GS_BACKEND`
- default fallback is `native_add3`
- log directory names include the backend suffix

## Practical Notes For Future Codex Work

- Treat this as a workspace-centric pipeline, not a library-first package.
- Be careful with auto-discovery when a workspace contains multiple depth or TAP outputs.
- Prefer preserving the YAML+CLI override pattern instead of hardcoding modes in Python.
- Many important artifacts are written under each scene workspace, not only at repo root.
- The repo may be in a dirty state during experiments; avoid reverting unrelated user changes.
- The most likely user-facing iteration points are profile YAMLs, `example.sh`, and the entrypoint scripts.

## Current Understanding Of The Demo Workflow

The intended user story appears to be:

1. Put frames under `WS/images` or provide a video.
2. Run `mosca_precompute.py` with a prep profile and optional backend overrides.
3. Run `mosca_reconstruct.py` with a fit profile.
4. Inspect `WS/logs/...` for bundle, scaffold, photometric outputs, metrics, and visualizations.

For camera-only experiments:

1. Precompute with `--skip_dynamic_resample`.
2. Run `lite_moca_reconstruct.py`.

This file should be the first context document to consult before changing configs, debugging a scene run, or modifying pipeline stages.
