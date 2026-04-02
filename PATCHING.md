# Patch Training

This document describes the current fixed-camera gen3c patch-training workflow implemented in:

- [mosca_patch.py](/home/atasoy/MoSca/mosca_patch.py)
- [lib_mosca/photo_recon.py](/home/atasoy/MoSca/lib_mosca/photo_recon.py)
- [data_utils/fixed_camera_sequence.py](/home/atasoy/MoSca/data_utils/fixed_camera_sequence.py)
- [render_experiment.py](/home/atasoy/MoSca/render_experiment.py)

## Goal

Patch an already trained MoSca `photometric` run with a second posed video source, while:

- keeping all cameras fixed
- preserving the original MoSca training set as the anchor supervision branch
- adding gen3c views as an extra RGB supervision branch
- allowing the scene representation to adapt to the new views

The intended use case is:

- existing trained MoSca model
- posed video-diffusion outputs with known intrinsics and poses
- optional normalization JSON that maps those gen3c cameras back to raw scale

## Current Assumptions

- Gen3c cameras are OpenCV `T_wc` cam-to-world poses.
- No similarity alignment is applied to cameras.
- No solved-space alignment is applied.
- The only camera-space transform applied to gen3c inputs is `normalization_params.json`.
- The default transform mode is `normalized_to_raw`.
- Gen3c frame index `i` is assumed to correspond to model time index `i`.
- If no gen3c priors exist, the gen3c branch uses RGB loss only.

## High-Level Flow

`mosca_patch.py` does the following:

1. Loads the base fit config and merges it with a patch config YAML.
2. Creates a new patch output directory under:
   - `<base_logdir>/patches/<patch_name>_<timestamp>/`
3. Loads the anchor MoSca state from the finished run:
   - `photometric_cam.pth`
   - `photometric_s_model_<backend>.pth`
   - `photometric_d_model_<backend>.pth`
   - `track_identification.npz`
   - the original `Saved2D` workspace data from `base_ws`
4. Loads the gen3c sequence through `FixedCameraRGBSequence`.
5. Optionally renders `preview_before`.
6. Runs `DynReconstructionSolver.patch_photometric_fit(...)`.
7. Saves patched checkpoints under the configured phase name, currently `patch_gen3c`.
8. Optionally renders `preview_after`.

## How To Launch

The patch entrypoint is:

```bash
python mosca_patch.py --cfg <patch_yaml>
```

Typical usage from the repo root:

```bash
source .venv/bin/activate
python mosca_patch.py --cfg profile/vipe/cowboy_cat_patch_gen3c.yaml
```

If needed, you can choose a device explicitly:

```bash
source .venv/bin/activate
python mosca_patch.py \
  --cfg profile/vipe/cowboy_cat_patch_gen3c.yaml \
  --device cuda:0
```

The patch run creates a fresh output directory under:

```text
<base_logdir>/patches/<patch_name>_<timestamp>/
```

So patching is non-destructive with respect to the original finished MoSca run.

## Launch In Tmux

For long runs, launch inside `tmux`.

Example:

```bash
tmux new-session -d -s cowboy_cat_gen3c_patch \
  'cd /home/atasoy/MoSca && \
   export MPLCONFIGDIR=/tmp/matplotlib-cowboy-cat-patch && \
   source /home/atasoy/MoSca/.venv/bin/activate && \
   python mosca_patch.py \
     --cfg /home/atasoy/MoSca/profile/vipe/cowboy_cat_patch_gen3c.yaml \
     > /home/atasoy/MoSca/runs/cowboy_cat_original_scale/logs/cowboy_cat_vipe_fixed_scene_native_add3_20260401_153850/patch_launch.log 2>&1'
```

Useful monitoring commands:

```bash
tail -f /home/atasoy/MoSca/runs/cowboy_cat_original_scale/logs/cowboy_cat_vipe_fixed_scene_native_add3_20260401_153850/patch_launch.log
```

```bash
ps -ef | grep '[p]ython mosca_patch.py'
```

## Required Patch Config Fields

At minimum, the patch YAML should define:

- `base_logdir`
- `base_ws`
- `base_fit_cfg` or a base run whose config can be inferred from `fit_commandline_args.txt`
- `gen3c_rgb_dir`
- `gen3c_pose_path`
- `gen3c_intrinsics_path`
- `gen3c_normalization_params`

Important optional fields:

- `patch_name`
- `patch_phase_name`
- `gen3c_normalization_mode`
- `gen3c_prior_ws`
- stage lengths and patch loss weights

## What The Config Controls

The merged patch config is:

- base MoSca fit config
- overridden or extended by the patch YAML

In practice this means:

- the base fit config still controls most original MoSca reconstruction losses and defaults
- the patch YAML adds patch-specific inputs and patch-specific optimization knobs

This is why `mosca_patch.py` takes only one user-facing `--cfg`: that YAML is merged with the base training config before patching starts.

## Gen3c Input Loading

The gen3c branch is represented by `FixedCameraRGBSequence`.

Inputs:

- `gen3c_rgb_dir`
- `gen3c_pose_path`
- `gen3c_intrinsics_path`
- `gen3c_normalization_params`
- `gen3c_normalization_mode`
- optional `gen3c_prior_ws`

Current behavior:

- Loads RGB frames from the directory in lexical order.
- Loads VIPE camera priors with `camera_convention="opencv"`.
- Applies `normalization_params.json` if provided.
- Stores:
  - `rgb`
  - `T_wc`
  - `K`
  - `model_tids`
  - `frame_names`

Validation currently enforced:

- RGB frames must exist and share one resolution.
- Pose count must match RGB frame count.
- `inds` must be contiguous `0..T-1`.
- Intrinsics must be valid with positive focal lengths.
- Principal point must lie inside the image.

Optional prior support:

- The class already has placeholders for future optional priors.
- Today, these hooks are not consumed by patch training.
- Even if `gen3c_prior_ws` is set, the current patch fit remains RGB-only on the gen3c branch.

## Optimization Structure

Patch fitting happens inside `DynReconstructionSolver.patch_photometric_fit(...)`.

Two supervision branches are mixed every training step:

- Anchor branch:
  - samples original MoSca views from the base workspace
  - uses the existing MoSca photometric and regularization logic
- Patch branch:
  - samples fixed gen3c views
  - renders the current scene from the supplied gen3c `T_wc`
  - compares render against gen3c RGB

By default:

- `anchor_views_per_step = 1`
- `patch_views_per_step = 1`
- `anchor_weight = 1.0`
- `patch_rgb_weight = 1.0`
- `patch_gen3c_rgb_ssim_lambda = 0.1`

Camera learning rates are forced to zero:

- `lr_cam_f = 0.0`
- `lr_cam_q = 0.0`
- `lr_cam_t = 0.0`

So the patch job does not optimize cameras.

## Stage Schedule

The current patch schedule is three-stage:

1. Stage 1: appearance-only stabilization
   - default `1000` steps
   - geometry-related learning rates are effectively held back
   - used to let the loaded model adapt appearance before moving structure
2. Stage 2: conservative geometry updates
   - default `2000` steps
   - geometry and scaffold/node learning rates are scaled down
   - defaults:
     - `stage2_geo_lr_scale = 0.25`
     - `stage2_node_lr_scale = 0.25`
     - `stage2_node_sigma_lr_scale = 0.25`
3. Stage 3: full patch refinement
   - default `1000` steps
   - geometry and scaffold/node learning rates return to full scale
   - defaults:
     - `stage3_geo_lr_scale = 1.0`
     - `stage3_node_lr_scale = 1.0`
     - `stage3_node_sigma_lr_scale = 1.0`

GS control is also active during patching, with separate static and dynamic settings for:

- densify
- prune
- opacity reset

## Losses

Anchor branch:

- keeps the original MoSca training losses and regularization active
- includes the same RGB, depth, track, temporal, distortion, ARAP, and velocity/acceleration terms configured by the merged fit config

Gen3c patch branch:

- currently RGB-only
- uses a photometric loss built from:
  - RGB reconstruction
  - SSIM mixing controlled by `patch_gen3c_rgb_ssim_lambda`

Not currently used on the gen3c branch:

- depth losses
- flow losses
- TAP / track losses
- prior masks

The code is intentionally structured so those can be added later without changing the fixed-camera patch architecture.

## Outputs

For `patch_phase_name: patch_gen3c`, the main saved outputs are:

- `patch_gen3c_cam.pth`
- `patch_gen3c_s_model_<backend>.pth`
- `patch_gen3c_d_model_<backend>.pth`
- `patch_gen3c_input_cameras.npz`
- `patch_gen3c_optim_loss.jpg`
- preview renders under:
  - `preview_before/`
  - `preview_after/`
- visualization videos such as:
  - `patch_gen3c_2dviz_0.mp4`
  - `patch_gen3c_3Dviz_0.mp4`

The patch output directory also stores:

- merged patch config
- command-line args
- dynamic reconstruction log

## Rendering Patched Checkpoints

`render_experiment.py` now supports alternate checkpoint families through:

- `--checkpoint_prefix`

Default behavior remains:

- `--checkpoint_prefix photometric`

Patched runs can now be rendered with:

- `--checkpoint_prefix patch_gen3c`

This makes the renderer load:

- `patch_gen3c_cam.pth`
- `patch_gen3c_s_model_<backend>.pth`
- `patch_gen3c_d_model_<backend>.pth`

Example:

```bash
python render_experiment.py \
  --logdir runs/.../patches/cowboy_cat_gen3c_patch_<timestamp> \
  --checkpoint_prefix patch_gen3c \
  --camera_set test \
  --region full \
  --test_camera_name gen3c_current_npz \
  --test_camera_pose_path gen3c-prior/.../pose/normalized_nofilter.npz \
  --test_camera_intrinsics_path gen3c-prior/.../intrinsics/normalized_nofilter.npz \
  --test_camera_normalization_params /path/to/normalization_params.json \
  --test_camera_normalization_mode normalized_to_raw
```

## Cowboy-Cat Reference Run

The current reference patch config is:

- [profile/vipe/cowboy_cat_patch_gen3c.yaml](/home/atasoy/MoSca/profile/vipe/cowboy_cat_patch_gen3c.yaml)

It patches this finished base run:

- [cowboy_cat_vipe_fixed_scene_native_add3_20260401_153850](/home/atasoy/MoSca/runs/cowboy_cat_original_scale/logs/cowboy_cat_vipe_fixed_scene_native_add3_20260401_153850)

Using these gen3c inputs:

- [pose/normalized_nofilter.npz](/home/atasoy/MoSca/gen3c-prior/cowboy-cat-zoom-in-out/gen3c/output/pose/normalized_nofilter.npz)
- [intrinsics/normalized_nofilter.npz](/home/atasoy/MoSca/gen3c-prior/cowboy-cat-zoom-in-out/gen3c/output/intrinsics/normalized_nofilter.npz)
- [rgb/frames](/home/atasoy/MoSca/gen3c-prior/cowboy-cat-zoom-in-out/gen3c/output/rgb/frames)

And this normalization JSON:

- [normalization_params.json](/home/atasoy/vipe/vipe_results/121frames/cowboy-cat/scene/normalized_nofilter/normalization_params.json)

The produced patch run is:

- [cowboy_cat_gen3c_patch_20260401_235730](/home/atasoy/MoSca/runs/cowboy_cat_original_scale/logs/cowboy_cat_vipe_fixed_scene_native_add3_20260401_153850/patches/cowboy_cat_gen3c_patch_20260401_235730)

The exact launch command used for that run was:

```bash
source /home/atasoy/MoSca/.venv/bin/activate
python /home/atasoy/MoSca/mosca_patch.py \
  --cfg /home/atasoy/MoSca/profile/vipe/cowboy_cat_patch_gen3c.yaml
```

And the `tmux` launch command used in practice was:

```bash
tmux new-session -d -s cowboy_cat_gen3c_patch \
  'cd /home/atasoy/MoSca && \
   export MPLCONFIGDIR=/tmp/matplotlib-cowboy-cat-patch && \
   source /home/atasoy/MoSca/.venv/bin/activate && \
   python mosca_patch.py \
     --cfg /home/atasoy/MoSca/profile/vipe/cowboy_cat_patch_gen3c.yaml \
     > /home/atasoy/MoSca/runs/cowboy_cat_original_scale/logs/cowboy_cat_vipe_fixed_scene_native_add3_20260401_153850/patch_launch.log 2>&1'
```

## Known Issue

The training run completed and saved patch checkpoints, but `mosca_patch.py` currently has a small post-run bug in `max_camera_state_delta(...)`.

Problem:

- it subtracts boolean tensors during the final camera-freeze check

Effect:

- training artifacts are still saved correctly
- the run crashes after training during final bookkeeping
- `camera_freeze_check.txt` is not written

This does not affect the actual patch checkpoints. In the reference run, the saved patched camera checkpoint was manually verified to be identical to the base camera checkpoint.
