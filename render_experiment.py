import argparse
import logging
import os
import os.path as osp
import shlex
import shutil
import subprocess

import imageio
import numpy as np
import torch
from matplotlib import cm
from omegaconf import OmegaConf

from data_utils.known_camera_helpers import (
    load_camera_normalization_params,
    load_vipe_camera_priors,
    transform_camera_T_wc_list,
)
from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian
from lib_render.render_helper import GS_BACKEND, render

"""
It's dumb but MoSca codebase uses "wc" for cam2world (world coordinates) as the naming.
"""


def configure_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser("Render a finished MoSca experiment")
    parser.add_argument("--logdir", type=str, required=True, help="Finished run dir")
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="photometric",
        help="Checkpoint prefix to load, e.g. photometric or fuse_gen3c",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Config used for training. Optional; used to visualize the input training cameras.",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default=None,
        help="Output directory. Defaults to <logdir>/renders",
    )
    parser.add_argument(
        "--camera_set",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Which camera trajectories to render",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="full",
        choices=["full", "static", "dynamic"],
        help="Which part of the reconstruction to render",
    )
    parser.add_argument("--fps", type=int, default=24, help="Output video fps")
    parser.add_argument(
        "--bg_color",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        help="Background color as RGB floats in [0,1]",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Torch device for rendering"
    )
    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Only save PNG frames, skip ffmpeg mp4/gif generation",
    )
    parser.add_argument(
        "--occupancy_threshold",
        type=float,
        default=1e-3,
        help="Alpha threshold used to convert the renderer's soft coverage into a binary occupancy mask",
    )
    parser.add_argument(
        "--test_camera_pose_path",
        type=str,
        default=None,
        help="Optional VIPE-style test-camera pose file (.npz or scene-style .txt)",
    )
    parser.add_argument(
        "--test_camera_intrinsics_path",
        type=str,
        default=None,
        help="Optional VIPE-style test-camera intrinsics file (.npz or scene-style .txt)",
    )
    parser.add_argument(
        "--test_camera_convention",
        type=str,
        default="opencv",
        choices=["opengl", "opencv"],
        help="Camera-axis convention of explicit test-camera poses",
    )
    parser.add_argument(
        "--test_camera_name",
        type=str,
        default="test_npz",
        help="Sequence name used when rendering explicit test-camera inputs",
    )
    parser.add_argument(
        "--test_height",
        type=int,
        default=None,
        help="Optional render height for explicit test-camera inputs. Defaults to training height",
    )
    parser.add_argument(
        "--test_width",
        type=int,
        default=None,
        help="Optional render width for explicit test-camera inputs. Defaults to training width",
    )
    parser.add_argument(
        "--test_camera_normalization_params",
        type=str,
        default=None,
        help="Optional VIPE normalization_params.json used to transform explicit test-camera poses",
    )
    parser.add_argument(
        "--test_camera_normalization_mode",
        type=str,
        default="normalized_to_raw",
        choices=["normalized_to_raw", "raw_to_normalized"],
        help="How to transform explicit test-camera poses with normalization_params.json",
    )
    parser.add_argument(
        "--show-cameras",
        action="store_true",
        help="Visualize camera sets with the same viewer used by mosca_reconstruct --show-cameras",
    )
    parser.add_argument(
        "--show-cameras-only",
        action="store_true",
        help="Open/save the camera visualization and skip RGB rendering",
    )
    parser.add_argument(
        "--camera-screenshot",
        type=str,
        default=None,
        help="Optional screenshot path for camera visualization. If set, runs the camera viewer off-screen",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=0.15,
        help="Relative camera-frustum scale passed to the shared camera viewer",
    )
    return parser.parse_args()


def load_cfg(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_readonly(cfg, True)
    return cfg


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


def resolve_cfg_path(args):
    if args.cfg is not None:
        return args.cfg
    return infer_cfg_from_logdir(args.logdir)


def load_experiment(logdir, device, checkpoint_prefix="photometric"):
    cams = MonocularCameras.load_from_ckpt(
        torch.load(
            osp.join(logdir, f"{checkpoint_prefix}_cam.pth"), map_location="cpu"
        )
    ).to(device)
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(logdir, f"{checkpoint_prefix}_s_model_{GS_BACKEND.lower()}.pth"),
            map_location="cpu",
        ),
        device=device,
    ).to(device)
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(logdir, f"{checkpoint_prefix}_d_model_{GS_BACKEND.lower()}.pth"),
            map_location="cpu",
        ),
        device=device,
    ).to(device)

    cams.eval()
    s_model.eval()
    d_model.eval()
    d_model.set_inference_mode()

    return cams, s_model, d_model


def build_intrinsics_from_focal(H, W, focal, cxcy_ratio, device):
    if torch.is_tensor(focal):
        focal = focal.detach().cpu().numpy()
    focal = np.asarray(focal, dtype=np.float32)
    if focal.ndim == 0:
        focal = np.array([float(focal), float(focal)], dtype=np.float32)
    if focal.size == 1:
        focal = np.array([float(focal[0]), float(focal[0])], dtype=np.float32)
    cxcy_ratio = np.asarray(cxcy_ratio, dtype=np.float32)
    L = min(H, W)
    K = torch.eye(3, device=device)
    K[0, 0] = float(focal[0]) * L / 2.0
    K[1, 1] = float(focal[1]) * L / 2.0
    K[0, 2] = float(W) * float(cxcy_ratio[0])
    K[1, 2] = float(H) * float(cxcy_ratio[1])
    return K


def build_intrinsics_from_K(K_src, device):
    K = torch.as_tensor(K_src, dtype=torch.float32, device=device).clone()
    assert K.shape == (3, 3) or (
        K.ndim == 3 and K.shape[1:] == (3, 3)
    ), f"Expected K shape (3,3) or (T,3,3), got {tuple(K.shape)}"
    return K


def validate_test_camera_normalization_args(args):
    if (
        args.test_camera_normalization_mode is not None
        and args.test_camera_normalization_params is None
    ):
        raise ValueError(
            "--test_camera_normalization_mode requires --test_camera_normalization_params"
        )
    if (
        args.test_camera_normalization_params is not None
        and args.test_camera_normalization_mode is None
    ):
        raise ValueError(
            "--test_camera_normalization_params requires --test_camera_normalization_mode"
        )


def convert_T_wc_to_T_cw(T_wc, device, dtype=torch.float32):
    T_wc = torch.as_tensor(T_wc, dtype=torch.float64).clone()
    assert T_wc.ndim == 3 and T_wc.shape[1:] == (
        4,
        4,
    ), f"Expected T_wc shape (T,4,4), got {tuple(T_wc.shape)}"

    R_wc = T_wc[:, :3, :3]
    t_wc = T_wc[:, :3, 3]
    R_cw = R_wc.transpose(1, 2)
    t_cw = -torch.einsum("tij,tj->ti", R_cw, t_wc)

    T_cw = torch.eye(4, dtype=torch.float64, device=T_wc.device)[None].repeat(
        len(T_wc), 1, 1
    )
    T_cw[:, :3, :3] = R_cw
    T_cw[:, :3, 3] = t_cw
    return T_cw.to(device=device, dtype=dtype)


def get_render_payload(region, s_model, d_model, tid):
    payload = []
    if region in ["full", "static"]:
        payload.append(s_model())
    if region in ["full", "dynamic"]:
        payload.append(d_model(tid))
    if len(payload) == 0:
        raise ValueError(f"Nothing to render for region={region}")
    return payload


def save_rgb_frame(save_fn, rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    imageio.imwrite(save_fn, (rgb * 255.0).astype(np.uint8))


def save_depth_frame(save_fn, depth):
    np.savez_compressed(save_fn, dep=depth.astype(np.float32))


def save_mask_frame(save_fn, mask):
    imageio.imwrite(save_fn, (mask.astype(np.uint8) * 255))


def colorize_depth_frames(depth_list, viz_quantile=3.0):
    dep_stack = np.stack(depth_list, axis=0).astype(np.float32)
    valid_mask = np.isfinite(dep_stack) & (dep_stack > 1e-6)

    if not np.any(valid_mask):
        return [np.zeros(depth.shape + (3,), dtype=np.uint8) for depth in depth_list]

    dep_values = dep_stack[valid_mask]
    dep_max = np.percentile(dep_values, 100.0 - viz_quantile)
    dep_min = np.percentile(dep_values, viz_quantile)
    if not np.isfinite(dep_min) or not np.isfinite(dep_max) or dep_max <= dep_min:
        dep_min = float(dep_values.min())
        dep_max = float(dep_values.max())

    denom = max(dep_max - dep_min, 1e-6)
    dep_norm = np.clip((dep_stack - dep_min) / denom, 0.0, 1.0)
    dep_norm[~valid_mask] = 0.0

    viz_list = []
    for dep in dep_norm:
        viz = cm.viridis(dep)[..., :3]
        viz_list.append((viz * 255.0).astype(np.uint8))
    return viz_list


def run_ffmpeg(cmd):
    logging.info("Running ffmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_video_outputs(frame_dir, fps, stem):
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg is required but was not found in PATH")

    frame_pattern = osp.join(frame_dir, "%05d.png")
    mp4_path = osp.join(osp.dirname(frame_dir), f"{stem}.mp4")
    gif_path = osp.join(osp.dirname(frame_dir), f"{stem}.gif")
    palette_path = osp.join(osp.dirname(frame_dir), f"{stem}_palette.png")

    run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-vf",
            "format=yuv420p",
            mp4_path,
        ]
    )
    run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-vf",
            "palettegen",
            palette_path,
        ]
    )
    run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-i",
            palette_path,
            "-lavfi",
            "paletteuse",
            gif_path,
        ]
    )
    if osp.exists(palette_path):
        os.remove(palette_path)


@torch.no_grad()
def render_sequence(
    save_root,
    sequence_name,
    H,
    W,
    K,
    T_wc_list,
    model_tids,
    s_model,
    d_model,
    region,
    bg_color,
    fps,
    skip_video,
    occupancy_threshold,
):
    os.makedirs(save_root, exist_ok=True)
    frame_dir = osp.join(save_root, "frames")
    depth_dir = osp.join(save_root, "depth")
    depth_viz_dir = osp.join(save_root, "depth_viz")
    occupancy_dir = osp.join(save_root, "occupancy")
    vace_mask_dir = osp.join(save_root, "vace_mask")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_viz_dir, exist_ok=True)
    os.makedirs(occupancy_dir, exist_ok=True)
    os.makedirs(vace_mask_dir, exist_ok=True)

    T_cw_list = convert_T_wc_to_T_cw(T_wc_list, device=K.device, dtype=K.dtype)
    if K.ndim == 3 and len(K) != len(T_cw_list):
        raise ValueError(
            f"Per-frame K length mismatch: {len(K)} intrinsics vs {len(T_cw_list)} poses"
        )
    depth_frames = []
    for frame_idx, (T_cw, model_tid) in enumerate(zip(T_cw_list, model_tids)):
        frame_K = K[frame_idx] if K.ndim == 3 else K
        render_dict = render(
            get_render_payload(region, s_model, d_model, int(model_tid)),
            H,
            W,
            frame_K,
            T_cw=T_cw,
            bg_color=bg_color,
        )
        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        depth = render_dict["dep"]
        if depth is None:
            raise RuntimeError(
                "The active renderer did not return depth. Depth export requires a "
                "backend that populates render_dict['dep']."
            )
        alpha = render_dict["alpha"]
        if alpha is None:
            raise RuntimeError(
                "The active renderer did not return alpha. Occupancy export requires a "
                "backend that populates render_dict['alpha']."
            )
        depth = depth.squeeze().detach().cpu().numpy()
        occupancy = (
            alpha.squeeze().detach().cpu().numpy() >= float(occupancy_threshold)
        )
        save_rgb_frame(osp.join(frame_dir, f"{frame_idx:05d}.png"), rgb)
        save_depth_frame(osp.join(depth_dir, f"{frame_idx:05d}.npz"), depth)
        save_mask_frame(osp.join(occupancy_dir, f"{frame_idx:05d}.png"), occupancy)
        save_mask_frame(osp.join(vace_mask_dir, f"{frame_idx:05d}.png"), ~occupancy)
        depth_frames.append(depth)

    depth_viz_frames = colorize_depth_frames(depth_frames)
    for frame_idx, depth_viz in enumerate(depth_viz_frames):
        imageio.imwrite(osp.join(depth_viz_dir, f"{frame_idx:05d}.png"), depth_viz)

    if not skip_video:
        build_video_outputs(frame_dir, fps=fps, stem=sequence_name)
        build_video_outputs(depth_viz_dir, fps=fps, stem=f"{sequence_name}_depth_viz")
        build_video_outputs(occupancy_dir, fps=fps, stem=f"{sequence_name}_occupancy")
        build_video_outputs(vace_mask_dir, fps=fps, stem="vace_mask")


def get_train_sequence(cams):
    H = int(cams.default_H)
    W = int(cams.default_W)
    K = cams.K(H, W)
    T_wc_list = cams.T_wc_list().detach()
    model_tids = list(range(cams.T))
    return [
        {
            "name": "train",
            "H": H,
            "W": W,
            "K": K,
            "T_wc_list": T_wc_list,
            "model_tids": model_tids,
        }
    ]


def get_initial_training_camera_set(cfg, cams):
    if cfg is None:
        return None

    known_camera_mode = getattr(cfg, "known_camera_mode", None)
    if known_camera_mode not in ["init", "fixed"]:
        return None

    known_camera_format = getattr(cfg, "known_camera_format", "vipe")
    if known_camera_format != "vipe":
        return None

    pose_path = getattr(cfg, "known_camera_pose_path", None)
    intr_path = getattr(cfg, "known_camera_intrinsics_path", None)
    if pose_path is None or intr_path is None:
        return None

    priors = load_vipe_camera_priors(
        pose_npz_path=pose_path,
        intrinsics_npz_path=intr_path,
        expected_T=int(cams.T),
        camera_convention=getattr(cfg, "known_camera_convention", "opencv"),
    )
    return {
        "name": "input training cameras",
        "T_wc": priors["T_wc"].detach().cpu().numpy(),
        "K": priors["K"].detach().cpu().numpy(),
        "color": "yellow",
    }


def get_test_sequences(args, device):
    if args.test_camera_pose_path is None or args.test_camera_intrinsics_path is None:
        return []

    test_priors = load_vipe_camera_priors(
        pose_npz_path=args.test_camera_pose_path,
        intrinsics_npz_path=args.test_camera_intrinsics_path,
        expected_T=None,
        camera_convention=getattr(args, "test_camera_convention", "opencv"),
    )
    test_T_wc = test_priors["T_wc"] # cam2world
    if args.test_camera_normalization_params is not None:
        normalization_params = load_camera_normalization_params(
            args.test_camera_normalization_params, device=test_T_wc.device
        )
        test_T_wc = transform_camera_T_wc_list(
            test_T_wc,
            normalization_params=normalization_params,
            mode=args.test_camera_normalization_mode,
        )
    test_inds = test_priors["inds"].detach().cpu().numpy().tolist()

    K = build_intrinsics_from_K(test_priors["K"], device=device)
    if args.test_height is not None and args.test_width is not None:
        H, W = int(args.test_height), int(args.test_width)
    else:
        # Infer H,W from the first principal point under the common cx=W/2, cy=H/2 convention.
        first_K = K[0] if K.ndim == 3 else K
        H, W = int(first_K[1, 2].item() * 2), int(first_K[0, 2].item() * 2)
    return [
        {
            "name": args.test_camera_name,
            "H": H,
            "W": W,
            "K": K,
            "T_wc_list": test_T_wc.to(device),
            "model_tids": [int(t) for t in test_inds],
        }
    ]


def get_test_camera_sets_for_visualization(args, cfg, cams, device):
    camera_sets = []

    explicit_sequences = get_test_sequences(args, device)
    for seq in explicit_sequences:
        camera_sets.append(
            {
                "name": seq["name"],
                "T_wc": seq["T_wc_list"].detach().cpu().numpy(),
                "K": seq["K"].detach().cpu().numpy(),
                "color": "red",
            }
        )
    return camera_sets


def maybe_visualize_render_cameras(args, cfg, cams, device):
    if not args.show_cameras:
        return

    try:
        from visualize_mosca_cameras import visualize_camera_sets
    except ImportError as exc:
        raise ImportError(
            "Camera visualization uses the same viewer as mosca_reconstruct --show-cameras, "
            "which requires visualize_mosca_cameras.py dependencies (notably pyvista and dreifus)."
        ) from exc

    camera_sets = []
    train_camera_set = get_initial_training_camera_set(cfg, cams)
    if train_camera_set is not None:
        camera_sets.append(train_camera_set)
    else:
        logging.warning("Could not determine initial training cameras from cfg")

    if args.camera_set in ["test", "both"]:
        camera_sets.extend(get_test_camera_sets_for_visualization(args, cfg, cams, device))

    if len(camera_sets) == 0:
        logging.warning("No camera sets available to visualize")
        return

    logging.info(
        "Showing render camera visualization. Close the window to continue."
    )
    visualize_camera_sets(
        camera_sets=camera_sets,
        stride=1,
        camera_scale=args.camera_scale,
        screenshot=args.camera_screenshot,
        title_lines=[
            "MoSca Cameras",
            "OpenCV camera convention; saved poses are T_wc (cam->world)",
            "yellow: initial training cameras",
            "red: render_experiment test cameras",
        ],
    )


def main():
    configure_logging()
    args = parse_args()
    validate_test_camera_normalization_args(args)
    device = torch.device(args.device)
    cfg_path = resolve_cfg_path(args)

    if args.savedir is None:
        savedir = osp.join(args.logdir, "renders")
    else:
        savedir = args.savedir
    os.makedirs(savedir, exist_ok=True)

    cams, s_model, d_model = load_experiment(
        args.logdir, device, checkpoint_prefix=args.checkpoint_prefix
    )

    cfg = None
    if args.camera_set in ["test", "both"] or args.show_cameras:
        if cfg_path is not None:
            cfg = load_cfg(cfg_path)
        elif (
            args.test_camera_pose_path is None
            or args.test_camera_intrinsics_path is None
        ):
            raise ValueError(
                "--cfg is required unless explicit test-camera pose and intrinsics inputs are provided"
            )

    maybe_visualize_render_cameras(args, cfg, cams, device)
    if args.show_cameras_only:
        logging.info("Finished camera visualization only.")
        return

    if args.camera_set in ["train", "both"]:
        for seq in get_train_sequence(cams):
            seq_root = osp.join(savedir, args.region, seq["name"])
            logging.info("Rendering %s to %s", seq["name"], seq_root)
            render_sequence(
                save_root=seq_root,
                sequence_name=seq["name"],
                H=seq["H"],
                W=seq["W"],
                K=seq["K"],
                T_wc_list=seq["T_wc_list"],
                model_tids=seq["model_tids"],
                s_model=s_model,
                d_model=d_model,
                region=args.region,
                bg_color=args.bg_color,
                fps=args.fps,
                skip_video=args.skip_video,
                occupancy_threshold=args.occupancy_threshold,
            )

    if args.camera_set in ["test", "both"]:
        for seq in get_test_sequences(args, device):
            seq_root = osp.join(savedir, args.region, seq["name"])
            logging.info("Rendering %s to %s", seq["name"], seq_root)
            render_sequence(
                save_root=seq_root,
                sequence_name=seq["name"],
                H=seq["H"],
                W=seq["W"],
                K=seq["K"],
                T_wc_list=seq["T_wc_list"],
                model_tids=seq["model_tids"],
                s_model=s_model,
                d_model=d_model,
                region=args.region,
                bg_color=args.bg_color,
                fps=args.fps,
                skip_video=args.skip_video,
                occupancy_threshold=args.occupancy_threshold,
            )

    logging.info("Finished rendering. Outputs saved to %s", savedir)


if __name__ == "__main__":
    main()
