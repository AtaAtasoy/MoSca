# CODEX Context

## What This Repository Is

MoSca is a monocular 4D reconstruction system with a two-stage operator workflow:

1. `mosca_precompute.py` builds 2D priors from a workspace of frames.
2. `mosca_reconstruct.py` solves camera/static geometry first, then dynamic scaffold + photometric reconstruction.

There is also a lighter camera-only path:

1. `mosca_precompute.py --skip_dynamic_resample`
2. `lite_moca_reconstruct.py`

The project is research-code style: config-heavy, GPU-first, and built around workspace folders that accumulate intermediate artifacts.

## Default Workflow

Use [example.sh](/home/atasoy/MoSca/example.sh) as the canonical run pattern.

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

- [mosca_precompute.py](/home/atasoy/MoSca/mosca_precompute.py): loads frames from `WS/images` or directly from an `.mp4`, runs depth/TAP/flow/EPI preprocessing, and optionally resamples more TAP tracks in dynamic regions.
- [lite_moca_reconstruct.py](/home/atasoy/MoSca/lite_moca_reconstruct.py): camera-focused reconstruction using MoCa bundle adjustment only.
- [mosca_reconstruct.py](/home/atasoy/MoSca/mosca_reconstruct.py): full reconstruction pipeline.
- [recon_utils.py](/home/atasoy/MoSca/recon_utils.py): workspace/log setup, auto-discovery of depth/TAP artifacts, track identification helpers.
- [example.sh](/home/atasoy/MoSca/example.sh): fastest way to understand intended operator usage.
- [readme.md](/home/atasoy/MoSca/readme.md): install notes, dataset modes, and paper-level context.

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

- [lib_prior/moca_processor.py](/home/atasoy/MoSca/lib_prior/moca_processor.py)
- [lib_prior/prior_loading.py](/home/atasoy/MoSca/lib_prior/prior_loading.py)

Supported prior backends visible in code:

- Depth: `metric3d`, `uni`, `depthcrafter`
- TAP: `spatracker`, `cotracker`, `bootstapir`
- Flow: `raft`

External-prior path now supported for VIPE-style inputs:

- `mosca_precompute.py` can import external depth priors instead of running a depth model when `known_depth_path` and `known_depth_format` are set.
- `mosca_precompute.py` can also use known intrinsics during preprocessing when `known_camera_intrinsics_path` is provided.
- `lite_moca_reconstruct.py` supports `known_camera_mode: init` with `known_camera_format: vipe`, initializing MoCa from external camera poses and intrinsics.
- The helper implementation lives in [data_utils/known_camera_helpers.py](/home/atasoy/MoSca/data_utils/known_camera_helpers.py).

### 2. Static Reconstruction

`lite_moca_reconstruct.static_reconstruct()`:

- loads RGB, depth, EPI, TAP, VOS-style masks through `Saved2D`
- auto-detects depth directory and TAP mode unless explicitly pinned in config
- runs `lib_moca.moca.moca_solve(...)`
- writes bundle-adjusted cameras and bundle state under `logs/.../bundle`

MoCa code lives mainly in:

- [lib_moca/moca.py](/home/atasoy/MoSca/lib_moca/moca.py)
- [lib_moca/camera.py](/home/atasoy/MoSca/lib_moca/camera.py)
- [lib_moca/bundle.py](/home/atasoy/MoSca/lib_moca/bundle.py)

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

- [lib_mosca/mosca.py](/home/atasoy/MoSca/lib_mosca/mosca.py)
- [lib_mosca/dynamic_solver.py](/home/atasoy/MoSca/lib_mosca/dynamic_solver.py)
- [lib_mosca/dynamic_solver_utils.py](/home/atasoy/MoSca/lib_mosca/dynamic_solver_utils.py)

### 5. Photometric Reconstruction

`mosca_reconstruct.photometric_reconstruct()`:

- loads cameras, scaffold, static/dynamic track IDs, flow, and priors
- builds static and dynamic Gaussian models
- runs joint photometric optimization
- emits the final reconstruction assets and evaluation outputs

Core files:

- [lib_mosca/photo_recon.py](/home/atasoy/MoSca/lib_mosca/photo_recon.py)
- [lib_mosca/static_gs.py](/home/atasoy/MoSca/lib_mosca/static_gs.py)
- [lib_mosca/dynamic_gs.py](/home/atasoy/MoSca/lib_mosca/dynamic_gs.py)
- [lib_mosca/photo_recon_utils.py](/home/atasoy/MoSca/lib_mosca/photo_recon_utils.py)

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

Profiles live under [profile](/home/atasoy/MoSca/profile):

- `profile/demo`: sample wild-scene settings used by `example.sh`
- `profile/iphone`
- `profile/nvidia`
- `profile/sintel`
- `profile/tum`
- `profile/vipe`: external-prior examples for VIPE camera/depth inputs

High-signal configs for first read:

- [profile/demo/demo_prep.yaml](/home/atasoy/MoSca/profile/demo/demo_prep.yaml)
- [profile/demo/demo_fit.yaml](/home/atasoy/MoSca/profile/demo/demo_fit.yaml)

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

## Modules By Responsibility

- [lib_prior](/home/atasoy/MoSca/lib_prior): foundational-model wrappers and preprocessing.
- [lib_moca](/home/atasoy/MoSca/lib_moca): camera/depth alignment and static bundle adjustment.
- [lib_mosca](/home/atasoy/MoSca/lib_mosca): dynamic scaffold + Gaussian reconstruction logic.
- [lib_render](/home/atasoy/MoSca/lib_render): Gaussian rendering backends and CUDA extensions.
- [data_utils](/home/atasoy/MoSca/data_utils): dataset-specific helpers.
- [eval_utils](/home/atasoy/MoSca/eval_utils): evaluation data and helpers.

If time is limited, read in this order:

1. [example.sh](/home/atasoy/MoSca/example.sh)
2. [profile/demo/demo_prep.yaml](/home/atasoy/MoSca/profile/demo/demo_prep.yaml)
3. [profile/demo/demo_fit.yaml](/home/atasoy/MoSca/profile/demo/demo_fit.yaml)
4. [mosca_precompute.py](/home/atasoy/MoSca/mosca_precompute.py)
5. [lite_moca_reconstruct.py](/home/atasoy/MoSca/lite_moca_reconstruct.py)
6. [mosca_reconstruct.py](/home/atasoy/MoSca/mosca_reconstruct.py)
7. [recon_utils.py](/home/atasoy/MoSca/recon_utils.py)

## Environment Assumptions

- Python 3.10 via conda in [install.sh](/home/atasoy/MoSca/install.sh)
- PyTorch 2.1.0 + CUDA 11.8 in the provided install path
- Linux/Ubuntu-oriented setup
- heavy GPU usage
- several third-party model weights expected under `weights/`

Rendering backend:

- [lib_render/render_helper.py](/home/atasoy/MoSca/lib_render/render_helper.py) reads `GS_BACKEND`
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
