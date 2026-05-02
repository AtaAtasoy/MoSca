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


def load_binary_masks_from_dir(mask_dir, frame_names):
    assert mask_dir is not None, "Expected a mask directory or stacked mask file path"

    if osp.isfile(mask_dir):
        if mask_dir.endswith(".npy"):
            masks = np.load(mask_dir)
        elif mask_dir.endswith(".npz"):
            mask_npz = np.load(mask_dir, allow_pickle=True)
            if "mask" in mask_npz.files:
                masks = mask_npz["mask"]
            elif "masks" in mask_npz.files:
                masks = mask_npz["masks"]
            else:
                raise ValueError(
                    f"Unsupported mask npz keys in {mask_dir}: {mask_npz.files}"
                )
        else:
            raise ValueError(f"Unsupported mask file: {mask_dir}")
        if masks.ndim == 2:
            masks = masks[None]
        assert (
            masks.ndim == 3
        ), f"Expected stacked mask shape (T,H,W), got {masks.shape}"
        if len(frame_names) != masks.shape[0]:
            raise ValueError(
                f"RGB/mask length mismatch: {len(frame_names)} RGB frames vs {masks.shape[0]} masks in {mask_dir}"
            )
        masks = masks > 0
        return torch.from_numpy(masks).bool()

    assert osp.isdir(mask_dir), f"Mask directory not found: {mask_dir}"
    mask_fns = [
        fn
        for fn in os.listdir(mask_dir)
        if fn.lower().endswith(".png") or fn.lower().endswith(".jpg")
    ]
    mask_fns = sorted(mask_fns)
    assert len(mask_fns) > 0, f"No mask frames found under {mask_dir}"
    if len(mask_fns) != len(frame_names):
        raise ValueError(
            f"RGB/mask length mismatch: {len(frame_names)} RGB frames vs {len(mask_fns)} masks under {mask_dir}"
        )

    masks = []
    for mask_fn in mask_fns:
        mask = imageio.imread(osp.join(mask_dir, mask_fn))
        if mask.ndim == 3:
            mask = mask[..., 0]
        masks.append(mask > 127)

    masks = np.stack(masks, 0)
    H, W = masks.shape[1:3]
    assert masks.ndim == 3, f"Expected mask frames shaped (T,H,W), got {masks.shape}"
    assert all(mask.shape == (H, W) for mask in masks), (
        "Mask frames must share one resolution"
    )
    return torch.from_numpy(masks).bool()


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
        dynamic_mask_dir=None,
    ):
        self.name = name
        self.rgb_dir = rgb_dir
        self.pose_path = pose_path
        self.intrinsics_path = intrinsics_path
        self.normalization_params_path = normalization_params_path
        self.normalization_mode = normalization_mode
        self.prior_ws = prior_ws
        self.dynamic_mask_dir = dynamic_mask_dir

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

        assert K.shape == (len(rgb), 3, 3), (
            f"Expected per-frame K shape ({len(rgb)},3,3), got {tuple(K.shape)}"
        )
        H, W = rgb.shape[1:3]
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]
        if torch.any(fx <= 0.0) or torch.any(fy <= 0.0):
            raise ValueError(
                f"Expected positive focal lengths, got fx range=({float(fx.min()):.3f}, {float(fx.max()):.3f}), "
                f"fy range=({float(fy.min()):.3f}, {float(fy.max()):.3f})"
            )
        if (
            torch.any(cx < 0.0)
            or torch.any(cx > float(W))
            or torch.any(cy < 0.0)
            or torch.any(cy > float(H))
        ):
            raise ValueError(
                f"Principal point must lie within the RGB resolution; got "
                f"cx range=({float(cx.min()):.3f}, {float(cx.max()):.3f}), "
                f"cy range=({float(cy.min()):.3f}, {float(cy.max()):.3f}) "
                f"for (W, H)=({W}, {H})"
            )

        if prior_ws is not None and not osp.isdir(prior_ws):
            raise FileNotFoundError(f"gen3c_prior_ws not found: {prior_ws}")

        if dynamic_mask_dir is not None:
            dynamic_masks = load_binary_masks_from_dir(dynamic_mask_dir, frame_names)
            if dynamic_masks.shape[1:3] != rgb.shape[1:3]:
                raise ValueError(
                    f"Mask/RGB resolution mismatch: {dynamic_masks.shape[1:3]} vs {rgb.shape[1:3]}"
                )
        else:
            dynamic_masks = None

        self.rgb = rgb
        self.frame_names = frame_names
        self.T_wc = T_wc.float()
        self.K = K.float()
        self.model_tids = inds.long()
        self.dynamic_masks = dynamic_masks
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
        if dynamic_mask_dir is not None:
            logging.info(
                "Loaded dynamic masks for fixed-camera RGB sequence '%s' from %s; fuse RGB loss will supervise only non-dynamic pixels",
                self.name,
                dynamic_mask_dir,
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
        if self.dynamic_masks is not None:
            self.dynamic_masks = self.dynamic_masks.to(device)
        for key, value in self.optional_priors.items():
            if torch.is_tensor(value):
                self.optional_priors[key] = value.to(device)
        return self

    def get_rgb_mask(self, index):
        if self.dynamic_masks is None:
            return torch.ones(self.H, self.W, dtype=torch.bool, device=self.rgb.device)
        # White pixels in the user-provided masks denote dynamic foreground, so
        # supervision keeps only the complementary static region.
        return ~self.dynamic_masks[index]

    def save_camera_npz(self, save_path):
        np.savez(
            save_path,
            T_wc=self.T_wc.detach().cpu().numpy(),
            K=self.K.detach().cpu().numpy(),
            inds=self.model_tids.detach().cpu().numpy(),
            frame_names=np.asarray(self.frame_names),
        )
