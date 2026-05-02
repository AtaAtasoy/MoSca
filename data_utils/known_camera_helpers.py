import json
import logging
import os.path as osp
from glob import glob

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _load_npz_data_and_inds(npz_path):
    assert npz_path is not None, "Expected a path to an .npz camera prior file"
    assert osp.exists(npz_path), f"Camera prior file not found: {npz_path}"
    data = np.load(npz_path, allow_pickle=True)
    assert "data" in data.files, f"Expected 'data' key in {npz_path}"
    inds = data["inds"] if "inds" in data.files else np.arange(len(data["data"]))
    return data["data"], inds


def _sort_by_inds(values, inds):
    order = np.argsort(inds)
    return values[order], np.asarray(inds)[order]


def _load_pose_data_and_inds(pose_path):
    assert pose_path is not None, "Expected a path to a VIPE pose prior file"
    assert osp.exists(pose_path), f"Camera prior file not found: {pose_path}"

    if pose_path.endswith(".npz"):
        pose_data, pose_inds = _load_npz_data_and_inds(pose_path)
        pose_data, pose_inds = _sort_by_inds(pose_data, pose_inds)
        assert pose_data.ndim == 3 and pose_data.shape[1:] == (
            4,
            4,
        ), f"Expected pose data shape (T,4,4), got {pose_data.shape}"
        return pose_data.astype(np.float32), pose_inds

    if pose_path.endswith(".txt"):
        pose_rows = np.loadtxt(pose_path, dtype=np.float64)
        pose_rows = np.atleast_2d(pose_rows)
        assert pose_rows.shape[1] == 8, (
            f"Expected pose text rows with 8 values "
            f"'frame_idx tx ty tz qw qx qy qz', got shape {pose_rows.shape}"
        )
        pose_inds = pose_rows[:, 0].round().astype(np.int64)
        pose_data = np.tile(np.eye(4, dtype=np.float32), (len(pose_rows), 1, 1))
        quat_xyzw = pose_rows[:, [5, 6, 7, 4]]
        pose_data[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix().astype(
            np.float32
        )
        pose_data[:, :3, 3] = pose_rows[:, 1:4].astype(np.float32)
        return _sort_by_inds(pose_data, pose_inds)

    raise ValueError(f"Unsupported VIPE pose prior file: {pose_path}")


def _load_intrinsics_data_and_inds(intrinsics_path):
    assert (
        intrinsics_path is not None
    ), "Expected a path to a VIPE intrinsics prior file"
    assert osp.exists(intrinsics_path), f"Camera prior file not found: {intrinsics_path}"

    if intrinsics_path.endswith(".npz"):
        intr_data, intr_inds = _load_npz_data_and_inds(intrinsics_path)
        intr_data, intr_inds = _sort_by_inds(intr_data, intr_inds)
        assert (
            intr_data.ndim == 2 and intr_data.shape[1] == 4
        ), f"Expected intrinsics data shape (T,4), got {intr_data.shape}"
        return intr_data.astype(np.float32), intr_inds

    if intrinsics_path.endswith(".txt"):
        intr_rows = np.loadtxt(intrinsics_path, dtype=np.float64)
        intr_rows = np.atleast_2d(intr_rows)
        if intr_rows.shape[1] == 9:
            fx = intr_rows[:, 0]
            fy = intr_rows[:, 4]
            cx = intr_rows[:, 2]
            cy = intr_rows[:, 5]
            intr_data = np.stack([fx, fy, cx, cy], axis=1)
        elif intr_rows.shape[1] == 4:
            intr_data = intr_rows
        else:
            raise AssertionError(
                "Expected intrinsics text rows with either 4 values "
                "'fx fy cx cy' or 9 flattened K values, got shape "
                f"{intr_rows.shape}"
            )
        intr_inds = np.arange(len(intr_data), dtype=np.int64)
        return intr_data.astype(np.float32), intr_inds

    raise ValueError(f"Unsupported VIPE intrinsics prior file: {intrinsics_path}")


def _convert_camera_pose_convention(T_wc_list, camera_convention):
    camera_convention = camera_convention.lower()
    if camera_convention in ["opencv", "open_cv"]:
        return T_wc_list.astype(np.float32)
    if camera_convention in ["opengl", "open_gl"]:
        # Convert a cam->world transform from OpenGL camera axes to OpenCV camera axes.
        # For c2w, this is a right-multiplication because we are changing the source
        # camera basis while keeping the world basis fixed.
        cv_from_gl = np.eye(4, dtype=np.float32)
        cv_from_gl[1, 1] = -1.0
        cv_from_gl[2, 2] = -1.0
        return (T_wc_list @ cv_from_gl[None]).astype(np.float32)
    raise ValueError(f"Unsupported camera_convention={camera_convention}")


def _intrinsics_rows_to_K(intr_data):
    intr_data = np.asarray(intr_data, dtype=np.float32)
    K = np.tile(np.eye(3, dtype=np.float32), (len(intr_data), 1, 1))
    K[:, 0, 0] = intr_data[:, 0]
    K[:, 1, 1] = intr_data[:, 1]
    K[:, 0, 2] = intr_data[:, 2]
    K[:, 1, 2] = intr_data[:, 3]
    return K


def load_vipe_camera_priors(
    pose_npz_path,
    intrinsics_npz_path,
    expected_T=None,
    intrinsics_tol=1e-4,
    camera_convention="opencv",
):
    pose_data, pose_inds = _load_pose_data_and_inds(pose_npz_path)
    intr_data, intr_inds = _load_intrinsics_data_and_inds(intrinsics_npz_path)

    assert len(pose_data) == len(
        intr_data
    ), f"Pose/intrinsics length mismatch: {len(pose_data)} vs {len(intr_data)}"
    assert np.array_equal(
        pose_inds, intr_inds
    ), "Pose and intrinsics frame indices do not match"

    if expected_T is not None:
        assert (
            len(pose_data) == expected_T
        ), f"Expected {expected_T} camera priors, got {len(pose_data)}"
        expected_inds = np.arange(expected_T)
        assert np.array_equal(
            pose_inds, expected_inds
        ), f"Expected contiguous frame indices 0..{expected_T-1}, got {pose_inds[:5]}...{pose_inds[-5:]}"

    intr_delta = np.abs(intr_data - intr_data[0][None]).max()
    K = _intrinsics_rows_to_K(intr_data)
    fx, fy, cx, cy = intr_data[0].tolist()

    logging.info(
        "Loaded VIPE camera priors from %s with T=%d, K_shape=%s, first fx=%.3f, fy=%.3f, cx=%.3f, cy=%.3f, intrinsics=%s, max_delta=%.6f, convention=%s",
        pose_npz_path,
        len(pose_data),
        K.shape,
        fx,
        fy,
        cx,
        cy,
        "shared" if intr_delta <= intrinsics_tol else "per-frame",
        intr_delta,
        camera_convention,
    )
    pose_data = _convert_camera_pose_convention(pose_data, camera_convention)
    return {
        "T_wc": torch.from_numpy(pose_data).float(),
        "K": torch.from_numpy(K).float(),
        "inds": torch.from_numpy(pose_inds).long(),
        "source": "vipe",
    }


def load_vipe_intrinsics_K(intrinsics_npz_path):
    intr_data, _ = _load_intrinsics_data_and_inds(intrinsics_npz_path)
    intr_delta = np.abs(intr_data - intr_data[0][None]).max()
    if intr_delta > 1e-4:
        logging.warning(
            "load_vipe_intrinsics_K returns the first K for legacy preprocessing compatibility; input intrinsics vary with max_delta=%.6f",
            intr_delta,
        )
    return _intrinsics_rows_to_K(intr_data)[0]


def load_vipe_depth_priors(scene_dir, expected_T=None):
    assert (
        scene_dir is not None
    ), "Expected a VIPE scene directory or stacked depth file path"

    depth_path = None
    dep_list = None

    if osp.isfile(scene_dir):
        if scene_dir.endswith(".npy"):
            depth_path = scene_dir
            dep_list = np.load(scene_dir).astype(np.float32)
        elif scene_dir.endswith(".npz"):
            depth_path = scene_dir
            depth_npz = np.load(scene_dir, allow_pickle=True)
            if "dep" in depth_npz.files:
                dep_list = depth_npz["dep"].astype(np.float32)
            elif "depth" in depth_npz.files:
                dep_list = depth_npz["depth"].astype(np.float32)
            else:
                raise ValueError(
                    f"Unsupported depth npz keys in {scene_dir}: {depth_npz.files}"
                )
        else:
            raise ValueError(f"Unsupported VIPE depth file: {scene_dir}")
    else:
        assert osp.isdir(scene_dir), f"VIPE scene directory not found: {scene_dir}"
        stacked_depth = osp.join(scene_dir, "depth.npy")
        if osp.exists(stacked_depth):
            depth_path = stacked_depth
            dep_list = np.load(stacked_depth).astype(np.float32)
        else:
            depth_fns = sorted(glob(osp.join(scene_dir, "frame_*.npy")))
            assert (
                len(depth_fns) > 0
            ), f"No depth.npy or frame_*.npy depth files found under {scene_dir}"
            if expected_T is not None:
                assert (
                    len(depth_fns) == expected_T
                ), f"Expected {expected_T} VIPE depth files, found {len(depth_fns)}"
            dep_list = np.stack([np.load(fn).astype(np.float32) for fn in depth_fns], 0)
            depth_path = scene_dir

    if dep_list.ndim == 2:
        dep_list = dep_list[None]
    assert (
        dep_list.ndim == 3
    ), f"Expected stacked VIPE depth shaped (T,H,W), got {dep_list.shape}"
    if expected_T is not None:
        assert (
            dep_list.shape[0] == expected_T
        ), f"Expected {expected_T} VIPE depth frames, found {dep_list.shape[0]}"

    logging.info(
        "Loaded VIPE depth priors from %s with T=%d, H=%d, W=%d",
        depth_path,
        dep_list.shape[0],
        dep_list.shape[1],
        dep_list.shape[2],
    )
    return dep_list


def load_camera_normalization_params(json_path, device=None):
    if not osp.exists(json_path):
        raise FileNotFoundError(f"Normalization params file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected normalization params JSON object, got {type(payload).__name__}"
        )
    if "center" not in payload or "scale" not in payload:
        raise ValueError(
            "Normalization params must contain 'center' and 'scale' keys"
        )

    if device is None:
        device = torch.device("cpu")
    center = torch.as_tensor(payload["center"], dtype=torch.float64, device=device)
    if center.shape != (3,):
        raise ValueError(
            f"Expected normalization center shape (3,), got {tuple(center.shape)}"
        )

    scale = float(payload["scale"])
    if scale <= 0.0:
        raise ValueError(f"Normalization scale must be positive, got {scale}")

    return {"center": center, "scale": scale}


def transform_camera_T_wc_list(T_wc, normalization_params, mode):
    if mode not in ["normalized_to_raw", "raw_to_normalized"]:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    T_wc = torch.as_tensor(T_wc, dtype=torch.float64).clone()
    assert T_wc.ndim == 3 and T_wc.shape[1:] == (
        4,
        4,
    ), f"Expected T_wc shape (T,4,4), got {tuple(T_wc.shape)}"

    center = normalization_params["center"].to(T_wc.device)
    scale = float(normalization_params["scale"])
    t_wc = T_wc[:, :3, 3]

    if mode == "normalized_to_raw":
        T_wc[:, :3, 3] = t_wc / scale + center[None]
    else:
        T_wc[:, :3, 3] = scale * (t_wc - center[None])
    T_wc[:, 3, :] = 0.0
    T_wc[:, 3, 3] = 1.0
    return T_wc
