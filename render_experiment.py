import argparse
import logging
import os
import os.path as osp
import shutil
import subprocess

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from data_utils.iphone_helpers import load_iphone_gt_poses
from data_utils.known_camera_helpers import load_vipe_camera_priors
from data_utils.nvidia_helpers import get_nvidia_dummy_test
from eval_utils.campose_alignment import align_ate_c2b_use_a2b
from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian
from lib_render.render_helper import GS_BACKEND, render


def configure_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser("Render a finished MoSca experiment")
    parser.add_argument("--logdir", type=str, required=True, help="Finished run dir")
    parser.add_argument(
        "--ws",
        type=str,
        default=None,
        help="Workspace root. Required for dataset-specific test cameras",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Config used for training. Required for test-camera rendering",
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
        default=[1.0, 1.0, 1.0],
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
        "--test_camera_pose_path",
        type=str,
        default=None,
        help="Optional .npz test-camera pose file with VIPE-style data/inds layout",
    )
    parser.add_argument(
        "--test_camera_intrinsics_path",
        type=str,
        default=None,
        help="Optional .npz test-camera intrinsics file with VIPE-style data/inds layout",
    )
    parser.add_argument(
        "--test_camera_convention",
        type=str,
        default="opencv",
        choices=["opengl", "opencv"],
        help="Camera-axis convention of explicit test-camera .npz poses",
    )
    parser.add_argument(
        "--test_camera_name",
        type=str,
        default="test_npz",
        help="Sequence name used when rendering explicit test-camera .npz inputs",
    )
    parser.add_argument(
        "--test_height",
        type=int,
        default=None,
        help="Optional render height for explicit test-camera .npz inputs. Defaults to training height",
    )
    parser.add_argument(
        "--test_width",
        type=int,
        default=None,
        help="Optional render width for explicit test-camera .npz inputs. Defaults to training width",
    )
    return parser.parse_args()


def load_cfg(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def get_known_camera_convention(cfg):
    if cfg is None:
        return "opencv"
    return getattr(cfg, "known_camera_convention", "opencv")


def load_experiment(logdir, device):
    cams = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(logdir, "photometric_cam.pth"), map_location="cpu")
    ).to(device)
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(logdir, f"photometric_s_model_{GS_BACKEND.lower()}.pth"),
            map_location="cpu",
        ),
        device=device,
    ).to(device)
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(logdir, f"photometric_d_model_{GS_BACKEND.lower()}.pth"),
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
    assert K.shape == (3, 3), f"Expected K shape (3,3), got {tuple(K.shape)}"
    return K


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
    T_cw_list,
    model_tids,
    s_model,
    d_model,
    region,
    bg_color,
    fps,
    skip_video,
):
    os.makedirs(save_root, exist_ok=True)
    frame_dir = osp.join(save_root, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    for frame_idx, (T_cw, model_tid) in enumerate(zip(T_cw_list, model_tids)):
        render_dict = render(
            get_render_payload(region, s_model, d_model, int(model_tid)),
            H,
            W,
            K,
            T_cw=T_cw,
            bg_color=bg_color,
        )
        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        save_rgb_frame(osp.join(frame_dir, f"{frame_idx:05d}.png"), rgb)

    if not skip_video:
        build_video_outputs(frame_dir, fps=fps, stem=sequence_name)


def get_train_sequence(cams):
    H = int(cams.default_H)
    W = int(cams.default_W)
    K = cams.K(H, W)
    T_cw_list = [cams.T_cw(t).detach() for t in range(cams.T)]
    model_tids = list(range(cams.T))
    return [
        {
            "name": "train",
            "H": H,
            "W": W,
            "K": K,
            "T_cw_list": T_cw_list,
            "model_tids": model_tids,
        }
    ]


def get_explicit_test_npz_sequences(args, device):
    if args.test_camera_pose_path is None or args.test_camera_intrinsics_path is None:
        return []

    test_priors = load_vipe_camera_priors(
        pose_npz_path=args.test_camera_pose_path,
        intrinsics_npz_path=args.test_camera_intrinsics_path,
        expected_T=None,
        camera_convention=getattr(args, "test_camera_convention", "opencv"),
    )
    test_inds = test_priors["inds"].detach().cpu().numpy().tolist()

    K = build_intrinsics_from_K(test_priors["K"], device=device)
    H, W = int(K[1, 2].item() * 2), int(K[0, 2].item() * 2) # infer H,W from cy,cx in K assuming cxcy_ratio=0.5
    test_cw_list = [test_priors["T_wc"][i].to(device) for i in range(len(test_priors["inds"]))] # cam2world, opencv coming from .npz
    return [
        {
            "name": args.test_camera_name,
            "H": H,
            "W": W,
            "K": K,
            "T_cw_list": test_cw_list,
            "model_tids": [int(t) for t in test_inds],
        }
    ]


def get_test_sequences(args, cfg, ws, cams, device):
    explicit_sequences = get_explicit_test_npz_sequences(args, device)
    if len(explicit_sequences) > 0:
        return explicit_sequences

    if cfg is None:
        raise ValueError(
            "--cfg is required for dataset-based test-camera rendering, or pass explicit --test_camera_pose_path and --test_camera_intrinsics_path"
        )
    if ws is None:
        raise ValueError("--ws is required for dataset-based test-camera rendering")

    dataset_mode = getattr(cfg, "mode", "iphone")
    solved_train_T_wi = cams.T_wc_list().detach().cpu()
    sequences = []

    if dataset_mode == "iphone":
        (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            _gt_testing_fns_list,
            _gt_training_fov,
            gt_testing_fov_list,
            _gt_training_cxcy_ratio,
            gt_testing_cxcy_ratio_list,
        ) = load_iphone_gt_poses(ws, getattr(cfg, "t_subsample", 1))

        test_image_dir = osp.join(ws, "test_images")
        sample_fn = sorted(
            [
                f
                for f in os.listdir(test_image_dir)
                if f.endswith(".png") or f.endswith(".jpg")
            ]
        )[0]
        sample = imageio.imread(osp.join(test_image_dir, sample_fn))
        H, W = sample.shape[:2]

        for cam_idx, test_cam_T_wi in enumerate(gt_testing_cam_T_wi_list):
            aligned_test_T_wi = align_ate_c2b_use_a2b(
                traj_a=gt_training_cam_T_wi,
                traj_b=solved_train_T_wi,
                traj_c=test_cam_T_wi,
            )
            T_cw_list = [
                torch.linalg.inv(aligned_test_T_wi[i]).to(device)
                for i in range(len(aligned_test_T_wi))
            ]
            focal = 1.0 / np.tan(np.deg2rad(gt_testing_fov_list[cam_idx]) / 2.0)
            K = build_intrinsics_from_focal(
                H=H,
                W=W,
                focal=focal,
                cxcy_ratio=gt_testing_cxcy_ratio_list[cam_idx],
                device=device,
            )
            sequences.append(
                {
                    "name": f"test_cam{cam_idx}",
                    "H": H,
                    "W": W,
                    "K": K,
                    "T_cw_list": T_cw_list,
                    "model_tids": [int(t) for t in gt_testing_tids_list[cam_idx]],
                }
            )

    elif dataset_mode == "nvidia":
        gt_training_cam_T_wi = solved_train_T_wi
        gt_training_fov = cams.fov
        (
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            _gt_testing_fns_list,
            gt_testing_fov_list,
            gt_testing_cxcy_ratio_list,
        ) = get_nvidia_dummy_test(gt_training_cam_T_wi, gt_training_fov)

        H = int(cams.default_H)
        W = int(cams.default_W)
        for cam_idx, test_cam_T_wi in enumerate(gt_testing_cam_T_wi_list):
            aligned_test_T_wi = align_ate_c2b_use_a2b(
                traj_a=gt_training_cam_T_wi,
                traj_b=solved_train_T_wi,
                traj_c=test_cam_T_wi,
            )
            T_cw_list = [
                torch.linalg.inv(aligned_test_T_wi[i]).to(device)
                for i in range(len(aligned_test_T_wi))
            ]
            focal = 1.0 / np.tan(np.deg2rad(float(gt_testing_fov_list[cam_idx])) / 2.0)
            K = build_intrinsics_from_focal(
                H=H,
                W=W,
                focal=focal,
                cxcy_ratio=gt_testing_cxcy_ratio_list[cam_idx],
                device=device,
            )
            sequences.append(
                {
                    "name": f"test_cam{cam_idx}",
                    "H": H,
                    "W": W,
                    "K": K,
                    "T_cw_list": T_cw_list,
                    "model_tids": [int(t) for t in gt_testing_tids_list[cam_idx]],
                }
            )
    else:
        raise ValueError(
            f"Test-camera rendering is only implemented for iphone/nvidia unless explicit .npz camera files are provided, got mode={dataset_mode}"
        )

    return sequences


def main():
    configure_logging()
    args = parse_args()
    device = torch.device(args.device)

    if args.savedir is None:
        savedir = osp.join(args.logdir, "renders")
    else:
        savedir = args.savedir
    os.makedirs(savedir, exist_ok=True)

    cams, s_model, d_model = load_experiment(args.logdir, device)

    cfg = None
    if args.camera_set in ["test", "both"]:
        if args.cfg is not None:
            cfg = load_cfg(args.cfg)
        elif (
            args.test_camera_pose_path is None
            or args.test_camera_intrinsics_path is None
        ):
            raise ValueError(
                "--cfg is required when rendering dataset test cameras unless explicit test .npz files are provided"
            )

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
                T_cw_list=seq["T_cw_list"],
                model_tids=seq["model_tids"],
                s_model=s_model,
                d_model=d_model,
                region=args.region,
                bg_color=args.bg_color,
                fps=args.fps,
                skip_video=args.skip_video,
            )

    if args.camera_set in ["test", "both"]:
        for seq in get_test_sequences(args, cfg, args.ws, cams, device):
            seq_root = osp.join(savedir, args.region, seq["name"])
            logging.info("Rendering %s to %s", seq["name"], seq_root)
            render_sequence(
                save_root=seq_root,
                sequence_name=seq["name"],
                H=seq["H"],
                W=seq["W"],
                K=seq["K"],
                T_cw_list=seq["T_cw_list"],
                model_tids=seq["model_tids"],
                s_model=s_model,
                d_model=d_model,
                region=args.region,
                bg_color=args.bg_color,
                fps=args.fps,
                skip_video=args.skip_video,
            )

    logging.info("Finished rendering. Outputs saved to %s", savedir)


if __name__ == "__main__":
    main()
