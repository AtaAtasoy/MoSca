import logging
import os.path as osp
from glob import glob

import numpy as np
import torch


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


def load_vipe_camera_priors(
    pose_npz_path,
    intrinsics_npz_path,
    expected_T=None,
    intrinsics_tol=1e-4,
):
    pose_data, pose_inds = _load_npz_data_and_inds(pose_npz_path)
    intr_data, intr_inds = _load_npz_data_and_inds(intrinsics_npz_path)

    pose_data, pose_inds = _sort_by_inds(pose_data, pose_inds)
    intr_data, intr_inds = _sort_by_inds(intr_data, intr_inds)

    assert pose_data.ndim == 3 and pose_data.shape[1:] == (
        4,
        4,
    ), f"Expected pose data shape (T,4,4), got {pose_data.shape}"
    assert (
        intr_data.ndim == 2 and intr_data.shape[1] == 4
    ), f"Expected intrinsics data shape (T,4), got {intr_data.shape}"
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

    intr_ref = intr_data[0]
    intr_delta = np.abs(intr_data - intr_ref[None]).max()
    if intr_delta > intrinsics_tol:
        raise ValueError(
            "VIPE intrinsics vary across time, but MonocularCameras currently expects "
            f"shared intrinsics. Max deviation was {intr_delta:.6f}."
        )

    fx, fy, cx, cy = intr_ref.tolist()
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    logging.info(
        "Loaded VIPE camera priors from %s with T=%d, fx=%.3f, fy=%.3f, cx=%.3f, cy=%.3f",
        pose_npz_path,
        len(pose_data),
        fx,
        fy,
        cx,
        cy,
    )
    return {
        "T_wc": torch.from_numpy(pose_data).float(),
        "K": torch.from_numpy(K).float(),
        "inds": torch.from_numpy(pose_inds).long(),
        "source": "vipe",
    }


def load_vipe_intrinsics_K(intrinsics_npz_path):
    intr_data, intr_inds = _load_npz_data_and_inds(intrinsics_npz_path)
    intr_data, _ = _sort_by_inds(intr_data, intr_inds)
    assert (
        intr_data.ndim == 2 and intr_data.shape[1] == 4
    ), f"Expected intrinsics data shape (T,4), got {intr_data.shape}"
    fx, fy, cx, cy = intr_data[0].tolist()
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def load_vipe_depth_priors(scene_dir, expected_T=None):
    assert scene_dir is not None, "Expected a directory containing VIPE scene depth .npy files"
    assert osp.isdir(scene_dir), f"VIPE scene directory not found: {scene_dir}"
    depth_fns = sorted(glob(osp.join(scene_dir, "frame_*.npy")))
    assert len(depth_fns) > 0, f"No frame_*.npy depth files found under {scene_dir}"
    if expected_T is not None:
        assert (
            len(depth_fns) == expected_T
        ), f"Expected {expected_T} VIPE depth files, found {len(depth_fns)}"
    dep_list = np.stack([np.load(fn).astype(np.float32) for fn in depth_fns], 0)
    logging.info(
        "Loaded VIPE depth priors from %s with T=%d, H=%d, W=%d",
        scene_dir,
        dep_list.shape[0],
        dep_list.shape[1],
        dep_list.shape[2],
    )
    return dep_list
