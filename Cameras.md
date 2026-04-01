# Cameras

This document summarizes camera I/O, the `MonocularCameras` class, the camera optimization flow, and the new path for initializing reconstruction from existing camera files.

## Scope

The main camera-related code paths are:

- `data_utils/known_camera_helpers.py`: external camera I/O for VIPE `.npz` and scene-style text exports
- `lib_moca/camera.py`: the `MonocularCameras` class
- `lite_moca_reconstruct.py`: static camera reconstruction entrypoint and external-init hook
- `lib_moca/moca.py`: MoCa camera initialization and handoff into BA
- `lib_moca/bundle.py`: static bundle adjustment
- `mosca_reconstruct.py`: later stages that reload, refine, and reuse cameras
- `lib_mosca/photo_recon.py`: final photometric save path

## Camera Data Contract

Across the repo, the canonical camera pose is `T_wc`, a 4x4 camera-to-world transform.

Important consequences:

- `T_wc` means points in camera coordinates are mapped into world coordinates.
- `T_cw` is always derived as `inverse(T_wc)`.
- External priors for the new path are expected in `T_wc` form as well.
- Visualization code also assumes saved poses are `T_wc`.

The canonical intrinsic representation in the repo is shared over time:

- `MonocularCameras` assumes one intrinsic parameter set for the whole sequence.
- Internally it stores:
  - relative focal on the short image side
  - principal point as `(cx / W, cy / H)`
- It does not store per-frame intrinsics.

That shared-intrinsics assumption is enforced explicitly in the VIPE loader. If imported intrinsics vary over time, loading fails instead of silently averaging them.

## External Camera I/O

The new external-camera initialization path is implemented in [`data_utils/known_camera_helpers.py`](/home/atasoy/MoSca/data_utils/known_camera_helpers.py#L38).

### Supported format

Current support is VIPE camera priors in two forms:

- legacy `.npz` files
- scene-style text exports with `pred_traj.txt` and `pred_intrinsics.txt`

Legacy `.npz` expectations:

- pose file:
  - required key: `data`
  - optional key: `inds`
  - expected shape: `(T, 4, 4)`
- intrinsics file:
  - required key: `data`
  - optional key: `inds`
  - expected shape: `(T, 4)` containing `fx, fy, cx, cy`

If `inds` is missing, the loader assumes `0..T-1`.

### Loader behavior

`load_vipe_camera_priors(...)` does the following:

1. Loads pose and intrinsics based on file extension.
2. Sorts both streams by frame index.
3. For text trajectories, parses `frame_idx tx ty tz qw qx qy qz` into OpenCV `T_wc`.
4. For text intrinsics, accepts either `fx fy cx cy` rows or flattened 3x3 `K` rows.
5. Checks that pose and intrinsics lengths match.
6. Checks that pose and intrinsics indices match exactly.
7. If `expected_T` is set, requires a contiguous `0..T-1` sequence.
8. Verifies intrinsics are effectively constant over time.
9. Builds one shared `K`.
10. Converts camera axes to OpenCV if the input convention is OpenGL.
11. Returns:
   - `T_wc`
   - `K`
   - `inds`
   - `source="vipe"`

See:

- [`data_utils/known_camera_helpers.py`](/home/atasoy/MoSca/data_utils/known_camera_helpers.py#L38)
- [`data_utils/known_camera_helpers.py`](/home/atasoy/MoSca/data_utils/known_camera_helpers.py#L23)

### Convention conversion

The loader accepts:

- `opencv`
- `opengl`

For OpenGL input, it right-multiplies each `T_wc` by a diagonal basis-change matrix that flips Y and Z. This keeps world coordinates fixed while changing the source camera basis into the OpenCV convention used by the rest of the repo.

That means a camera import bug is very likely to be one of these:

- wrong assumption about whether the source poses are `cam->world` or `world->cam`
- wrong `known_camera_convention`
- non-contiguous or mismatched `inds`
- time-varying intrinsics in a system that assumes shared intrinsics

## `MonocularCameras`

The project-wide camera object is [`lib_moca/camera.py`](/home/atasoy/MoSca/lib_moca/camera.py#L16).

### Intrinsics

Initialization accepts either:

- `fxfycxcy=[fov_x_deg, fov_y_deg, cx_ratio, cy_ratio]`, or
- `K=<3x3 intrinsic matrix>`

Internally, `K` is converted into:

- `_rel_focal`
- `cxcy_ratio`

See:

- [`lib_moca/camera.py`](/home/atasoy/MoSca/lib_moca/camera.py#L88)
- [`lib_moca/camera.py`](/home/atasoy/MoSca/lib_moca/camera.py#L275)

`K(H, W)` reconstructs pixel intrinsics from the relative representation using `L = min(H, W)`.

### Extrinsics

The class supports two extrinsic parameterizations:

1. Delta mode
2. Independent mode

In delta mode:

- frame 0 is stored as a full world pose
- later frames are stored as relative transforms
- `forward_T()` accumulates them left-to-right

See:

- [`lib_moca/camera.py`](/home/atasoy/MoSca/lib_moca/camera.py#L112)
- [`lib_moca/camera.py`](/home/atasoy/MoSca/lib_moca/camera.py#L137)

In independent mode:

- each frame directly stores its own `T_wc`

See:

- [`lib_moca/camera.py`](/home/atasoy/MoSca/lib_moca/camera.py#L155)
- [`lib_moca/camera.py`](/home/atasoy/MoSca/lib_moca/camera.py#L184)

### Important detail about delta mode

When `delta_flag=True`, `init_camera_pose` is not a full `T_wc` list. It is expected to contain `T-1` relative transforms. When `delta_flag=False`, `init_camera_pose` is expected to be the full `T_wc` list.

That distinction matters a lot for debugging imports:

- external VIPE initialization uses `delta_flag=False` because the imported poses are absolute `T_wc`
- internal FOV-based initialization uses `delta_flag=True` because it starts from pairwise relative motion

### Projection model

Projection and backprojection operate in normalized image coordinates defined on the short side of the image, not in raw pixels.

Core helpers:

- `project(xyz)`
- `backproject(uv, d)`
- `trans_pts_to_world(tid, pts_c)`
- `trans_pts_to_cam(tid, pts_w)`

See:

- [`lib_moca/camera.py`](/home/atasoy/MoSca/lib_moca/camera.py#L320)
- [`lib_moca/camera.py`](/home/atasoy/MoSca/lib_moca/camera.py#L357)

This is another easy place to get confused during debugging: some code works in pixel `K`, while BA math is performed using normalized homogeneous image coordinates from `Saved2D`.

### Checkpoint format

`MonocularCameras.load_from_ckpt(...)` reconstructs the module from a saved state dict, including backward compatibility for older key names.

See:

- [`lib_moca/camera.py`](/home/atasoy/MoSca/lib_moca/camera.py#L61)

## Static Camera Creation And Optimization Flow

The main camera creation path starts in [`lite_moca_reconstruct.py`](/home/atasoy/MoSca/lite_moca_reconstruct.py#L144).

### Step 1: load workspace priors into `Saved2D`

`static_reconstruct(...)` builds `Saved2D` by loading:

- EPI
- depth
- normalized depth
- recomputed depth mask
- uniform TAP tracks
- VOS

See:

- [`lite_moca_reconstruct.py`](/home/atasoy/MoSca/lite_moca_reconstruct.py#L156)

This means camera optimization always happens against already-normalized depth, not raw imported depth.

### Step 2: choose the camera initialization source

`static_reconstruct(...)` chooses one of three paths:

1. external known-camera init
2. dataset GT init
3. no init, so MoCa estimates focal and motion itself

See:

- [`lite_moca_reconstruct.py`](/home/atasoy/MoSca/lite_moca_reconstruct.py#L169)

#### Path A: external known-camera init

`load_known_camera_init(...)`:

- reads VIPE priors
- builds a `MonocularCameras` with:
  - `K=priors["K"]`
  - `delta_flag=False`
  - `init_camera_pose=priors["T_wc"]`

See:

- [`lite_moca_reconstruct.py`](/home/atasoy/MoSca/lite_moca_reconstruct.py#L55)

This is the new path that loads existing cameras as initialization.

#### Path B: dataset GT init

If `init_gt_camera` is enabled, GT cameras are loaded from the dataset helpers and converted into `MonocularCameras`.

There are two variants:

- focal-only GT init:
  - initializes focal from GT
  - initializes delta extrinsics from identity relative poses
- full GT init:
  - initializes full absolute poses with `delta_flag=False`

See:

- [`lite_moca_reconstruct.py`](/home/atasoy/MoSca/lite_moca_reconstruct.py#L172)

#### Path C: no prior camera

If no camera object is provided, `moca_solve(...)` estimates:

- a focal value by search
- pairwise relative motion from static tracks
- an initial delta-mode camera

See:

- [`lib_moca/moca.py`](/home/atasoy/MoSca/lib_moca/moca.py#L199)

### Step 3: static track identification

`moca_solve(...)` identifies static tracks from:

- precomputed RAFT EPI if available, or
- track-based epipolar analysis otherwise

Only those static tracks feed camera initialization and BA.

See:

- [`lib_moca/moca.py`](/home/atasoy/MoSca/lib_moca/moca.py#L140)

### Step 4: optional focal search and internal camera creation

If no camera prior is provided:

- MoCa computes an optimal focal by search
- computes neighboring relative motions
- builds a delta-mode `MonocularCameras`

See:

- [`lib_moca/moca.py`](/home/atasoy/MoSca/lib_moca/moca.py#L223)
- [`lib_moca/moca.py`](/home/atasoy/MoSca/lib_moca/moca.py#L253)

If a prior camera is provided:

- MoCa skips this internal initialization path
- uses the passed camera object directly
- optionally rescales its translation to match normalized depth

See:

- [`lib_moca/moca.py`](/home/atasoy/MoSca/lib_moca/moca.py#L266)
- [`lib_moca/moca.py`](/home/atasoy/MoSca/lib_moca/moca.py#L27)

That translation rescaling is especially important when imported poses and imported depth are in different metric scales.

### Step 5: static bundle adjustment

`compute_static_ba(...)` jointly optimizes:

- camera rotation
- camera translation
- camera focal
- per-frame depth scale
- optional depth correction

See:

- [`lib_moca/bundle.py`](/home/atasoy/MoSca/lib_moca/bundle.py#L19)
- [`lib_moca/bundle.py`](/home/atasoy/MoSca/lib_moca/bundle.py#L93)

Loss terms include:

- reprojection / flow consistency
- depth consistency
- optional camera smoothness

See:

- [`lib_moca/bundle.py`](/home/atasoy/MoSca/lib_moca/bundle.py#L198)
- [`lib_moca/bundle.py`](/home/atasoy/MoSca/lib_moca/bundle.py#L225)

If the camera starts in delta mode, BA switches it to independent mode at `switch_to_ind_step`.

See:

- [`lib_moca/bundle.py`](/home/atasoy/MoSca/lib_moca/bundle.py#L117)
- [`lib_moca/camera.py`](/home/atasoy/MoSca/lib_moca/camera.py#L221)

### Step 6: camera checkpoint save

At the end of static BA:

- cameras are saved to `bundle/bundle_cams.pth`
- depth scale and correction are saved to `bundle/bundle.pth`

See:

- [`lib_moca/bundle.py`](/home/atasoy/MoSca/lib_moca/bundle.py#L300)

## New Existing-Camera Initialization Path

The new path is configured in profiles like [`profile/vipe/cowboy_cat_fit_init.yaml`](/home/atasoy/MoSca/profile/vipe/cowboy_cat_fit_init.yaml).

Relevant fields:

- `known_camera_mode: init`
- `known_camera_format: vipe`
- `known_camera_convention: opencv` or `opengl`
- `known_camera_pose_path`
- `known_camera_intrinsics_path`

### What it actually does

When those fields are set:

1. `lite_moca_reconstruct.static_reconstruct(...)` creates `Saved2D`.
2. `load_known_camera_init(...)` loads the external poses and intrinsics.
3. It instantiates `MonocularCameras` in independent mode from the imported absolute `T_wc`.
4. That camera object is passed to `moca_solve(...)` as `gt_cam`.
5. `moca_solve(...)` skips focal-search initialization and uses the imported camera as the optimization starting point.
6. BA still refines the camera unless the relevant learning rates are zero.

See:

- [`lite_moca_reconstruct.py`](/home/atasoy/MoSca/lite_moca_reconstruct.py#L55)
- [`lite_moca_reconstruct.py`](/home/atasoy/MoSca/lite_moca_reconstruct.py#L214)
- [`lib_moca/moca.py`](/home/atasoy/MoSca/lib_moca/moca.py#L266)

### Practical implication

This path is initialization, not a hard constraint.

If you want to keep imported cameras mostly fixed, the main knobs are:

- `ba_lr_cam_q`
- `ba_lr_cam_t`
- `ba_lr_cam_f`

The checked-in VIPE profile already sets `ba_lr_cam_f: 0.0`, which freezes focal but still allows pose refinement.

See:

- [`profile/vipe/cowboy_cat_fit_init.yaml`](/home/atasoy/MoSca/profile/vipe/cowboy_cat_fit_init.yaml)

## Camera Lifecycle Across The Full Pipeline

There are three main camera checkpoints to keep in mind:

### 1. Static BA output

`bundle/bundle_cams.pth`

Written by static BA.

See:

- [`lib_moca/bundle.py`](/home/atasoy/MoSca/lib_moca/bundle.py#L300)

### 2. Warmup-updated bundle camera

`bundle/bundle_cams.pth`

If photometric warmup is enabled, the original BA camera is renamed to `bundle/bundle_cams_ba.pth`, and the refined warmup camera is written back to `bundle/bundle_cams.pth`.

See:

- [`mosca_reconstruct.py`](/home/atasoy/MoSca/mosca_reconstruct.py#L173)
- [`mosca_reconstruct.py`](/home/atasoy/MoSca/mosca_reconstruct.py#L181)

### 3. Final photometric camera

`photometric_cam.pth`

The final photometric fitting phase saves `f"{phase_name}_cam.pth"`, and the default phase name is `photometric`.

See:

- [`lib_mosca/photo_recon.py`](/home/atasoy/MoSca/lib_mosca/photo_recon.py#L420)
- [`lib_mosca/photo_recon.py`](/home/atasoy/MoSca/lib_mosca/photo_recon.py#L1138)

When debugging camera outputs, make sure you are looking at the right stage. A mismatch between `bundle/bundle_cams.pth` and `photometric_cam.pth` can be completely expected.

## Precompute-Time Camera I/O

There is also camera-related I/O during preprocessing.

`mosca_precompute.py` can load known intrinsics and pass them into `MoCaPrep`:

- during depth processing
- during TAP computation when the backend uses intrinsics

See:

- [`mosca_precompute.py`](/home/atasoy/MoSca/mosca_precompute.py#L71)
- [`mosca_precompute.py`](/home/atasoy/MoSca/mosca_precompute.py#L102)
- [`lib_prior/moca_processor.py`](/home/atasoy/MoSca/lib_prior/moca_processor.py#L618)

This path only imports intrinsics, not poses.

## Visualization And Rendering Paths

The camera debugging utilities already understand the new VIPE input path.

### `visualize_mosca_cameras.py`

Can show:

- input training cameras from config-provided VIPE priors
- bundle cameras
- final optimized cameras
- explicit test camera `.npz` trajectories

It treats imported and optimized poses as `T_wc`.

### `render_experiment.py`

Loads the solved training cameras from `photometric_cam.pth`, and for explicit test-camera rendering it:

- loads train priors if configured
- aligns test poses through the train-prior trajectory into solved-camera coordinates
- renders using the aligned `T_cw`

This means rendering debug issues may come from alignment, not only from raw pose import.

## VIPE Faithful Mode

The repo now supports a fixed-geometry path for external VIPE scenes:

- `known_scene_mode: fixed`
- `known_camera_mode: fixed`
- `known_depth_mode: fixed`

In that mode:

- imported VIPE poses remain fixed through BA, warmup, and photometric fitting
- imported VIPE intrinsics remain fixed
- imported VIPE depth values are preserved instead of being globally normalized
- bundle depth replay is skipped in later stages
- unit-sensitive world thresholds are internally rescaled so existing normalized-scene configs remain usable

The checkpoint contract stays the same:

- `bundle/bundle_cams.pth`
- `bundle/bundle.pth`
- `photometric_cam.pth`

But in faithful mode those artifacts should reflect fixed scene geometry rather than optimized camera/depth geometry.

## Debugging Checklist

When camera I/O looks wrong, check these in order:

1. Confirm the source pose convention.
   - The repo expects `T_wc`.
   - If the source file is actually `T_cw`, everything will look mirrored or inverted.

2. Confirm the camera axis convention.
   - `known_camera_convention=opencv` vs `opengl` is a real behavioral switch.

3. Confirm frame indices.
   - VIPE import with `expected_T=s2d.T` requires contiguous `0..T-1`.

4. Confirm intrinsics are constant over time.
   - The current camera class cannot represent time-varying intrinsics.

5. Confirm the workspace image size matches the imported intrinsics.
   - `MonocularCameras` converts `K` using `default_H/default_W`.

6. Confirm whether BA is allowed to move the imported camera.
   - Nonzero `ba_lr_cam_q` and `ba_lr_cam_t` mean imported poses are only an initialization.

7. Confirm depth scale compatibility.
   - Imported poses may need `rescale_gt_cam_transl=True` if depth and pose scales differ.

8. Confirm which checkpoint you are inspecting.
   - `bundle/bundle_cams.pth`
   - `bundle/bundle_cams_ba.pth`
   - `photometric_cam.pth`

## Most Likely Failure Modes

Based on the current implementation, the highest-signal camera bugs are:

- importing `T_cw` when the code expects `T_wc`
- wrong OpenGL/OpenCV convention selection
- sequence length or `inds` mismatch
- trying to use time-varying intrinsics with `MonocularCameras`
- expecting imported cameras to remain fixed even though BA is optimizing them
- comparing against the wrong output checkpoint after warmup or photometric refinement
