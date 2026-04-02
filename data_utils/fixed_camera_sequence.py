import logging
import os
import os.path as osp

import imageio
import numpy as np
import torch

from data_utils.known_camera_helpers import (
    load_camera_normalization_params,
    load_vipe_camera_priors,
    transform_camera_T_wc_list,
)


def load_rgb_frames_from_dir(rgb_dir):
    assert osp.isdir(rgb_dir), f"RGB directory not found: {rgb_dir}"
    img_fns = sorted(
        [
            fn
            for fn in os.listdir(rgb_dir)
            if fn.lower().endswith(".png") or fn.lower().endswith(".jpg")
        ]
    )
    assert len(img_fns) > 0, f"No RGB frames found under {rgb_dir}"

    images = []
    for fn in img_fns:
        img = imageio.imread(osp.join(rgb_dir, fn))[..., :3]
        images.append(img)
    images = np.stack(images, 0)

    H, W = images.shape[1:3]
    assert (
        images.ndim == 4 and images.shape[-1] == 3
    ), f"Expected RGB frames shaped (T,H,W,3), got {images.shape}"
    assert all(img.shape[:2] == (H, W) for img in images), "RGB frames must share one resolution"

    frame_names = [osp.splitext(fn)[0] for fn in img_fns]
    rgb = torch.from_numpy(images).float() / 255.0
    return rgb, frame_names


class FixedCameraRGBSequence:
    def __init__(
        self,
        name,
        rgb_dir,
        pose_path,
        intrinsics_path,
        normalization_params_path=None,
        normalization_mode="normalized_to_raw",
        prior_ws=None,
    ):
        self.name = name
        self.rgb_dir = rgb_dir
        self.pose_path = pose_path
        self.intrinsics_path = intrinsics_path
        self.normalization_params_path = normalization_params_path
        self.normalization_mode = normalization_mode
        self.prior_ws = prior_ws

        rgb, frame_names = load_rgb_frames_from_dir(rgb_dir)
        priors = load_vipe_camera_priors(
            pose_npz_path=pose_path,
            intrinsics_npz_path=intrinsics_path,
            expected_T=len(rgb),
            camera_convention="opencv",
        )
        T_wc = priors["T_wc"].clone()
        if normalization_params_path is not None:
            normalization_params = load_camera_normalization_params(
                normalization_params_path, device=T_wc.device
            )
            T_wc = transform_camera_T_wc_list(
                T_wc,
                normalization_params=normalization_params,
                mode=normalization_mode,
            ).float()

        K = priors["K"].clone()
        inds = priors["inds"].clone()
        assert torch.equal(
            inds, torch.arange(len(inds), dtype=inds.dtype)
        ), f"Expected contiguous frame inds 0..T-1, got {inds[:5]}...{inds[-5:]}"
        assert len(rgb) == len(T_wc), f"RGB/camera length mismatch: {len(rgb)} vs {len(T_wc)}"

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        H, W = rgb.shape[1:3]
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError(f"Expected positive focal lengths, got fx={fx}, fy={fy}")
        if not (0.0 <= cx <= float(W) and 0.0 <= cy <= float(H)):
            raise ValueError(
                f"Principal point must lie within the RGB resolution; got "
                f"(cx, cy)=({cx:.3f}, {cy:.3f}) for (W, H)=({W}, {H})"
            )

        if prior_ws is not None and not osp.isdir(prior_ws):
            raise FileNotFoundError(f"gen3c_prior_ws not found: {prior_ws}")

        self.rgb = rgb
        self.frame_names = frame_names
        self.T_wc = T_wc.float()
        self.K = K.float()
        self.model_tids = inds.long()
        self.optional_priors = {}
        self.has_optional_priors = False

        logging.info(
            "Loaded fixed-camera RGB sequence '%s' with T=%d, H=%d, W=%d from %s",
            self.name,
            self.T,
            self.H,
            self.W,
            rgb_dir,
        )
        if prior_ws is not None:
            logging.info(
                "Optional prior workspace is configured at %s but RGB-only losses remain active until prior hooks are enabled",
                prior_ws,
            )

    @property
    def T(self):
        return int(self.rgb.shape[0])

    @property
    def H(self):
        return int(self.rgb.shape[1])

    @property
    def W(self):
        return int(self.rgb.shape[2])

    def to(self, device):
        self.rgb = self.rgb.to(device)
        self.T_wc = self.T_wc.to(device)
        self.K = self.K.to(device)
        self.model_tids = self.model_tids.to(device)
        for key, value in self.optional_priors.items():
            if torch.is_tensor(value):
                self.optional_priors[key] = value.to(device)
        return self

    def get_rgb_mask(self, index):
        return torch.ones(self.H, self.W, dtype=torch.bool, device=self.rgb.device)

    def save_camera_npz(self, save_path):
        np.savez(
            save_path,
            T_wc=self.T_wc.detach().cpu().numpy(),
            K=self.K.detach().cpu().numpy(),
            inds=self.model_tids.detach().cpu().numpy(),
            frame_names=np.asarray(self.frame_names),
        )
