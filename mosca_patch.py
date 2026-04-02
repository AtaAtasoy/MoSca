import argparse
from datetime import datetime
import logging
import os
import os.path as osp
import shlex

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from data_utils.fixed_camera_sequence import FixedCameraRGBSequence
from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.photo_recon import (
    DynReconstructionSolver,
    OptimCFG,
    GSControlCFG,
    convert_single_T_wc_to_T_cw,
)
from lib_mosca.static_gs import StaticGaussian
from lib_prior.prior_loading import Saved2D
from lib_render.render_helper import GS_BACKEND, render
from recon_utils import (
    SEED,
    auto_get_depth_dir_tap_mode,
    log_geometry_policy,
    seed_everything,
)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser("Patch an existing MoSca run with fixed gen3c cameras")
    parser.add_argument("--cfg", type=str, required=True, help="Patch config YAML")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Torch device for patch fitting"
    )
    return parser.parse_args()


def _read_fit_commandline_args(logdir):
    args_path = osp.join(logdir, "fit_commandline_args.txt")
    if not osp.exists(args_path):
        return []
    with open(args_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if len(content) == 0:
        return []
    return shlex.split(content)


def _extract_cli_flag(tokens, flag):
    for i, token in enumerate(tokens):
        if token == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def infer_cfg_from_logdir(logdir):
    tokens = _read_fit_commandline_args(logdir)
    return _extract_cli_flag(tokens, "--cfg")


def load_patch_cfg(cfg_path):
    patch_cfg = OmegaConf.load(cfg_path)
    base_logdir = getattr(patch_cfg, "base_logdir", None)
    if base_logdir is None:
        raise ValueError("Patch config must provide base_logdir")

    base_fit_cfg_path = getattr(patch_cfg, "base_fit_cfg", None)
    if base_fit_cfg_path is None:
        base_fit_cfg_path = infer_cfg_from_logdir(base_logdir)
    if base_fit_cfg_path is None:
        raise ValueError(
            "Could not infer the base fit config from base_logdir; set base_fit_cfg explicitly"
        )

    base_fit_cfg = OmegaConf.load(base_fit_cfg_path)
    merged_cfg = OmegaConf.merge(base_fit_cfg, patch_cfg)
    return merged_cfg, base_fit_cfg_path


def setup_patch_logdir(base_logdir, cfg):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    patch_name = getattr(cfg, "patch_name", "gen3c_patch")
    patch_root = osp.join(base_logdir, "patches")
    patch_logdir = osp.join(patch_root, f"{patch_name}_{now}")
    os.makedirs(patch_logdir, exist_ok=True)
    with open(osp.join(patch_logdir, "patch_commandline_args.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(os.sys.argv))
    OmegaConf.save(config=cfg, f=osp.join(patch_logdir, "patch_cfg_merged.yaml"))
    logging.info("Patch logdir: %s", patch_logdir)
    return patch_logdir


def load_anchor_patch_state(base_ws, base_logdir, fit_cfg, patch_logdir, device):
    policy = log_geometry_policy(fit_cfg, prefix="Patch anchor geometry policy")
    depth_dir, tap_mode = auto_get_depth_dir_tap_mode(base_ws, fit_cfg)
    depth_boundary_th = getattr(fit_cfg, "depth_boundary_th", 1.0)
    dep_median = (
        getattr(fit_cfg, "dep_median", 1.0)
        if policy["apply_depth_normalization"]
        else -1.0
    )

    s2d = (
        Saved2D(base_ws)
        .load_epi()
        .load_dep(depth_dir, depth_boundary_th)
        .normalize_depth(
            median_depth=dep_median,
            apply_scale=policy["apply_depth_normalization"],
        )
        .recompute_dep_mask(depth_boundary_th=depth_boundary_th)
        .load_track(
            tap_mode, min_valid_cnt=getattr(fit_cfg, "tap_loading_min_valid_cnt", 4)
        )
        .load_vos()
        .load_flow()
        .to(device)
    )
    if policy["replay_bundle_depth"]:
        s2d = s2d.rescale_perframe_depth_from_bundle(
            bundle_pth_fn=osp.join(base_logdir, "bundle", "bundle.pth")
        )

    track_identification = np.load(osp.join(base_logdir, "track_identification.npz"))
    s2d.register_track_indentification(
        torch.from_numpy(track_identification["static_track_mask"]).to(device),
        torch.from_numpy(track_identification["dynamic_track_mask"]).to(device),
    )

    cam_path = osp.join(base_logdir, "photometric_cam.pth")
    if not osp.exists(cam_path):
        cam_path = osp.join(base_logdir, "bundle", "bundle_cams.pth")
    cams = MonocularCameras.load_from_ckpt(
        torch.load(cam_path, map_location="cpu")
    ).to(device)

    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(base_logdir, f"photometric_s_model_{GS_BACKEND.lower()}.pth"),
            map_location="cpu",
        ),
        device=device,
    ).to(device)

    d_model_path = osp.join(base_logdir, f"photometric_d_model_{GS_BACKEND.lower()}.pth")
    if osp.exists(d_model_path):
        d_model = DynSCFGaussian.load_from_ckpt(
            torch.load(d_model_path, map_location="cpu"),
            device=device,
        ).to(device)
        d_model.scf.set_multi_level(
            mlevel_arap_flag=True,
            mlevel_list=getattr(fit_cfg, "photo_mlevel_list", [1, 6]),
            mlevel_k_list=getattr(fit_cfg, "photo_mlevel_k_list", [16, 8]),
            mlevel_w_list=getattr(fit_cfg, "photo_mlevel_w_list", [0.4, 0.3]),
        )
    else:
        logging.warning("No dynamic checkpoint found at %s; patch static-only", d_model_path)
        d_model = None

    photo_solver = DynReconstructionSolver(
        working_dir=patch_logdir,
        device=device,
        radius_init_factor=getattr(fit_cfg, "gs_radius_init_factor", 4.0),
        opacity_init_factor=getattr(fit_cfg, "gs_opacity_init_factor", 0.95),
    )
    photo_solver.identify_fg_mask_by_nearest_curve(
        s2d, cams, viz_fname="patch_anchor_fg_mask.mp4"
    )
    if GS_BACKEND == "gof" and (
        getattr(fit_cfg, "photo_lambda_normal", 0.0) > 0.0
        or getattr(fit_cfg, "photo_lambda_depth_normal", 0.0) > 0.0
    ):
        photo_solver.compute_normals_for_s2d(
            s2d,
            cams,
            patch_size=7,
            nn_dist_th=0.03,
            nn_min_cnt=4,
            viz_fn="patch_anchor_normal.mp4",
        )
    return s2d, cams, s_model, d_model, photo_solver


@torch.no_grad()
def render_patch_preview(save_root, prefix, sequence, s_model, d_model, bg_color):
    os.makedirs(save_root, exist_ok=True)
    if sequence.T == 0:
        return
    preview_inds = sorted(set([0, sequence.T // 2, sequence.T - 1]))
    for seq_index in preview_inds:
        model_tid = int(sequence.model_tids[seq_index].item())
        gs5 = [s_model()]
        if d_model is not None:
            gs5.append(d_model(model_tid))
        render_dict = render(
            gs5,
            sequence.H,
            sequence.W,
            sequence.K,
            convert_single_T_wc_to_T_cw(sequence.T_wc[seq_index]),
            bg_color=bg_color,
        )
        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        save_fn = osp.join(save_root, f"{prefix}_{seq_index:05d}.png")
        imageio.imwrite(save_fn, np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8))


def max_camera_state_delta(before_state, after_state):
    max_delta = 0.0
    for key, before_value in before_state.items():
        after_value = after_state[key]
        if torch.is_tensor(before_value):
            delta = float(
                (before_value - after_value.detach().cpu()).abs().max().item()
            )
            max_delta = max(max_delta, delta)
    return max_delta


def main():
    configure_logging()
    args = parse_args()
    cfg, base_fit_cfg_path = load_patch_cfg(args.cfg)
    seed_everything(SEED)

    base_logdir = getattr(cfg, "base_logdir")
    base_ws = getattr(cfg, "base_ws", None)
    if base_ws is None:
        raise ValueError("Patch config must provide base_ws")

    device = torch.device(args.device)
    patch_logdir = setup_patch_logdir(base_logdir, cfg)
    with open(osp.join(patch_logdir, "base_fit_cfg_path.txt"), "w", encoding="utf-8") as f:
        f.write(base_fit_cfg_path)

    patch_seq = FixedCameraRGBSequence(
        name=getattr(cfg, "patch_sequence_name", "gen3c"),
        rgb_dir=getattr(cfg, "gen3c_rgb_dir"),
        pose_path=getattr(cfg, "gen3c_pose_path"),
        intrinsics_path=getattr(cfg, "gen3c_intrinsics_path"),
        normalization_params_path=getattr(cfg, "gen3c_normalization_params"),
        normalization_mode=getattr(cfg, "gen3c_normalization_mode", "normalized_to_raw"),
        prior_ws=getattr(cfg, "gen3c_prior_ws", None),
    ).to(device)
    patch_seq.save_camera_npz(osp.join(patch_logdir, "patch_gen3c_input_cameras.npz"))

    anchor_s2d, anchor_cams, s_model, d_model, photo_solver = load_anchor_patch_state(
        base_ws=base_ws,
        base_logdir=base_logdir,
        fit_cfg=cfg,
        patch_logdir=patch_logdir,
        device=device,
    )

    base_cam_state = {
        key: value.detach().cpu().clone()
        for key, value in anchor_cams.state_dict().items()
        if torch.is_tensor(value)
    }

    if getattr(cfg, "patch_save_preview_renders", True):
        render_patch_preview(
            osp.join(patch_logdir, "preview_before"),
            "gen3c_before",
            patch_seq,
            s_model,
            d_model,
            bg_color=getattr(cfg, "photo_default_bg_color", [1.0, 1.0, 1.0]),
        )

    photo_solver.patch_photometric_fit(
        anchor_s2d=anchor_s2d,
        anchor_cams=anchor_cams,
        patch_seq=patch_seq,
        s_model=s_model,
        d_model=d_model,
        total_steps=getattr(
            cfg,
            "patch_total_steps",
            getattr(cfg, "patch_stage1_steps", 1000)
            + getattr(cfg, "patch_stage2_steps", 2000)
            + getattr(cfg, "patch_stage3_steps", 1000),
        ),
        stage1_steps=getattr(cfg, "patch_stage1_steps", 1000),
        stage2_steps=getattr(cfg, "patch_stage2_steps", 2000),
        stage3_steps=getattr(cfg, "patch_stage3_steps", 1000),
        anchor_views_per_step=getattr(cfg, "patch_anchor_views_per_step", 1),
        patch_views_per_step=getattr(cfg, "patch_gen3c_views_per_step", 1),
        topo_update_feq=getattr(cfg, "patch_topo_update_feq", 50),
        skinning_corr_start_steps=getattr(
            cfg, "patch_skinning_corr_start_steps", 10000000000
        ),
        s_gs_ctrl_cfg=GSControlCFG(
            densify_steps=getattr(cfg, "patch_s_ctrl_densify_steps", 400),
            reset_steps=getattr(cfg, "patch_s_ctrl_reset_steps", 1000),
            prune_steps=getattr(cfg, "patch_s_ctrl_prune_steps", 400),
            densify_max_grad=getattr(cfg, "patch_s_ctrl_densify_max_grad", 0.0002),
            densify_percent_dense=getattr(
                cfg, "patch_s_ctrl_densify_percent_dense", 0.01
            ),
            prune_opacity_th=getattr(cfg, "patch_s_ctrl_prune_opacity_th", 0.02),
            reset_opacity=getattr(cfg, "patch_s_ctrl_reset_opacity", 0.01),
        ),
        d_gs_ctrl_cfg=GSControlCFG(
            densify_steps=getattr(cfg, "patch_d_ctrl_densify_steps", 300),
            reset_steps=getattr(cfg, "patch_d_ctrl_reset_steps", 1000),
            prune_steps=getattr(cfg, "patch_d_ctrl_prune_steps", 300),
            densify_max_grad=getattr(cfg, "patch_d_ctrl_densify_max_grad", 0.0001),
            densify_percent_dense=getattr(
                cfg, "patch_d_ctrl_densify_percent_dense", 0.01
            ),
            prune_opacity_th=getattr(cfg, "patch_d_ctrl_prune_opacity_th", 0.02),
            reset_opacity=getattr(cfg, "patch_d_ctrl_reset_opacity", 0.01),
        ),
        optimizer_cfg=OptimCFG(
            lr_cam_f=0.0,
            lr_cam_q=0.0,
            lr_cam_t=0.0,
            lr_p=getattr(cfg, "patch_lr_p", getattr(cfg, "photo_lr_p", 0.00016)),
            lr_q=getattr(cfg, "patch_lr_q", getattr(cfg, "photo_lr_q", 0.001)),
            lr_s=getattr(cfg, "patch_lr_s", getattr(cfg, "photo_lr_s", 0.005)),
            lr_o=getattr(cfg, "patch_lr_o", getattr(cfg, "photo_lr_o", 0.05)),
            lr_sph=getattr(cfg, "patch_lr_sph", getattr(cfg, "photo_lr_sph", 0.0025)),
            lr_sph_rest_factor=getattr(
                cfg,
                "patch_lr_sph_rest_factor",
                getattr(cfg, "photo_lr_sph_rest_factor", 20.0),
            ),
            lr_p_final=getattr(
                cfg,
                "patch_lr_p_final",
                getattr(cfg, "photo_lr_p_final", 0.00016 / 100),
            ),
            lr_np=getattr(cfg, "patch_lr_np", getattr(cfg, "photo_lr_np", 0.0001)),
            lr_nq=getattr(cfg, "patch_lr_nq", getattr(cfg, "photo_lr_nq", 0.001)),
            lr_nsig=getattr(
                cfg, "patch_lr_nsig", getattr(cfg, "photo_lr_nsig", 0.01)
            ),
            lr_np_final=getattr(
                cfg,
                "patch_lr_np_final",
                getattr(cfg, "photo_lr_np_final", 0.0000016),
            ),
            lr_nq_final=getattr(
                cfg,
                "patch_lr_nq_final",
                getattr(cfg, "photo_lr_nq_final", 0.00001),
            ),
            lr_w=getattr(cfg, "patch_lr_w", 0.0),
            lr_w_final=getattr(cfg, "patch_lr_w_final", None),
        ),
        anchor_weight=getattr(cfg, "patch_anchor_weight", 1.0),
        patch_rgb_weight=getattr(cfg, "patch_gen3c_rgb_weight", 1.0),
        patch_rgb_ssim_lambda=getattr(cfg, "patch_gen3c_rgb_ssim_lambda", 0.1),
        lambda_rgb=getattr(cfg, "photo_lambda_rgb", 1.0),
        lambda_dep=getattr(cfg, "photo_lambda_dep", 0.01),
        lambda_mask=getattr(cfg, "photo_lambda_mask", 0.0),
        dep_st_invariant=getattr(cfg, "photo_dep_st_invariant", True),
        lambda_normal=getattr(cfg, "photo_lambda_normal", 0.1),
        lambda_depth_normal=getattr(cfg, "photo_lambda_depth_normal", 0.05),
        lambda_distortion=getattr(cfg, "photo_lambda_distortion", 100.0),
        lambda_arap_coord=getattr(cfg, "photo_lambda_arap_coord", 3.0),
        lambda_arap_len=getattr(cfg, "photo_lambda_arap_len", 3.0),
        lambda_vel_xyz_reg=getattr(cfg, "photo_lambda_vel_xyz_reg", 3.0),
        lambda_vel_rot_reg=getattr(cfg, "photo_lambda_vel_rot_reg", 1.0),
        lambda_acc_xyz_reg=getattr(cfg, "photo_lambda_acc_xyz_reg", 3.0),
        lambda_acc_rot_reg=getattr(cfg, "photo_lambda_acc_rot_reg", 3.0),
        lambda_small_w_reg=getattr(cfg, "photo_lambda_small_w_reg", 0.01),
        lambda_track=getattr(cfg, "photo_lambda_track", 0.01),
        track_flow_chance=getattr(cfg, "photo_track_flow_chance", 0.5),
        track_flow_interval_candidates=getattr(
            cfg, "photo_track_flow_interval_candidates", [1, 3]
        ),
        track_loss_interval=getattr(cfg, "photo_track_loss_interval", 4),
        track_loss_start_step=getattr(cfg, "photo_track_loss_start_step", -1),
        track_loss_end_step=getattr(cfg, "photo_track_loss_end_step", 6000),
        reg_radius=getattr(cfg, "patch_reg_radius", None),
        use_decay=getattr(cfg, "patch_use_decay", False),
        decay_start=getattr(cfg, "patch_decay_start", 2000),
        temporal_diff_shift=getattr(cfg, "photo_temporal_diff_shift", [1, 3, 8]),
        temporal_diff_weight=getattr(cfg, "photo_temporal_diff_weight", [0.6, 0.2, 0.2]),
        geo_reg_start_steps=getattr(cfg, "patch_geo_reg_start_steps", 0),
        dyn_scf_prune_steps=getattr(cfg, "patch_dyn_scf_prune_steps", []),
        dyn_scf_prune_sk_th=getattr(cfg, "patch_dyn_scf_prune_sk_th", 0.02),
        stage2_geo_lr_scale=getattr(cfg, "patch_stage2_geo_lr_scale", 0.25),
        stage2_node_lr_scale=getattr(cfg, "patch_stage2_node_lr_scale", 0.25),
        stage2_node_sigma_lr_scale=getattr(
            cfg, "patch_stage2_node_sigma_lr_scale", 0.25
        ),
        stage3_geo_lr_scale=getattr(cfg, "patch_stage3_geo_lr_scale", 1.0),
        stage3_node_lr_scale=getattr(cfg, "patch_stage3_node_lr_scale", 1.0),
        stage3_node_sigma_lr_scale=getattr(
            cfg, "patch_stage3_node_sigma_lr_scale", 1.0
        ),
        viz_skip_t=getattr(cfg, "patch_viz_skip_t", 1),
        viz_cheap_interval=getattr(cfg, "patch_viz_cheap_interval", 500),
        viz_move_angle_deg=getattr(cfg, "patch_viz_move_angle_deg", 10.0),
        random_bg=getattr(cfg, "photo_random_bg", True),
        default_bg_color=getattr(cfg, "photo_default_bg_color", [0.0, 0.0, 0.0]),
        phase_name=getattr(cfg, "patch_phase_name", "patch_gen3c"),
    )

    if getattr(cfg, "patch_save_preview_renders", True):
        render_patch_preview(
            osp.join(patch_logdir, "preview_after"),
            "gen3c_after",
            patch_seq,
            s_model,
            d_model,
            bg_color=getattr(cfg, "photo_default_bg_color", [1.0, 1.0, 1.0]),
        )

    camera_delta = max_camera_state_delta(
        base_cam_state, anchor_cams.state_dict()
    )
    with open(osp.join(patch_logdir, "camera_freeze_check.txt"), "w", encoding="utf-8") as f:
        f.write(f"max_camera_state_delta={camera_delta:.10f}\n")
    logging.info("Fixed-camera patch finished with max_camera_state_delta=%.10f", camera_delta)


if __name__ == "__main__":
    main()
