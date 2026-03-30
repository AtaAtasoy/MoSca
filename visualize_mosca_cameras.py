import argparse
import logging
import os
import os.path as osp
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from data_utils.iphone_helpers import load_iphone_gt_poses
from data_utils.known_camera_helpers import load_vipe_camera_priors
from data_utils.nvidia_helpers import load_nvidia_gt_pose, get_nvidia_dummy_test
from eval_utils.campose_alignment import align_ate_c2b_use_a2b
from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import SH2RGB, StaticGaussian
from lib_render.render_helper import GS_BACKEND
import pyvista as pv
from dreifus.matrix import (
    CameraCoordinateConvention,
    Intrinsics,
    Pose,
    PoseType,
)
from dreifus.pyvista import add_camera_frustum, add_coordinate_axes

def configure_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MoSca input / optimized / test cameras with Dreifus + PyVista"
    )
    parser.add_argument("--logdir", type=str, required=True, help="Finished run dir")
    parser.add_argument("--cfg", type=str, default=None, help="Training config")
    parser.add_argument("--ws", type=str, default=None, help="Workspace root")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device for loading Gaussian data"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Show every N-th camera for each sequence"
    )
    parser.add_argument(
        "--show_input_cameras",
        action="store_true",
        help="Visualize input training cameras if they can be found",
    )
    parser.add_argument(
        "--align_input_cameras_to_optimized",
        action="store_true",
        help="Align input training cameras to the optimized trajectory before plotting",
    )
    parser.add_argument(
        "--show_optimized_cameras",
        action="store_true",
        help="Visualize final optimized cameras from photometric_cam.pth",
    )
    parser.add_argument(
        "--show_bundle_cameras",
        action="store_true",
        help="Visualize bundle cameras from bundle/bundle_cams.pth if present",
    )
    parser.add_argument(
        "--show_test_cameras",
        action="store_true",
        help="Visualize dataset or explicit test-camera trajectories",
    )
    parser.add_argument(
        "--test_camera_pose_path",
        type=str,
        default=None,
        help="Optional explicit VIPE-style test pose .npz file",
    )
    parser.add_argument(
        "--test_camera_intrinsics_path",
        type=str,
        default=None,
        help="Optional explicit VIPE-style test intrinsics .npz file",
    )
    parser.add_argument(
        "--test_camera_convention",
        type=str,
        default="opencv",
        choices=["opengl", "opencv"],
        help="Camera-axis convention of explicit test-camera .npz poses",
    )
    parser.add_argument(
        "--show_static_points",
        action="store_true",
        help="Show static Gaussian centers as context",
    )
    parser.add_argument(
        "--show_dynamic_points",
        action="store_true",
        help="Show dynamic Gaussian centers at one timestep as context",
    )
    parser.add_argument(
        "--dynamic_time",
        type=int,
        default=0,
        help="Timestep used when visualizing dynamic Gaussian centers",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=3.0,
        help="PyVista point size for Gaussian-center visualization",
    )
    parser.add_argument(
        "--camera_scale",
        type=float,
        default=0.15,
        help="Relative camera frustum scale passed to Dreifus",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Optional screenshot output path. If set, renders off-screen instead of opening a window",
    )
    return parser.parse_args()


def load_cfg(cfg_path: Optional[str]):
    if cfg_path is None:
        return None
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def load_camera_checkpoint(cam_path: str, device: torch.device):
    cams = MonocularCameras.load_from_ckpt(
        torch.load(cam_path, map_location="cpu")
    ).to(device)
    cams.eval()
    return cams


def load_optimized_cameras(logdir: str, device: torch.device):
    return load_camera_checkpoint(osp.join(logdir, "photometric_cam.pth"), device)


def load_bundle_cameras(logdir: str, device: torch.device):
    bundle_path = osp.join(logdir, "bundle", "bundle_cams.pth")
    if not osp.exists(bundle_path):
        return None
    return load_camera_checkpoint(bundle_path, device)


def intrinsics_from_K(K_np, Intrinsics):
    return Intrinsics(
        float(K_np[0, 0]),
        float(K_np[1, 1]),
        float(K_np[0, 2]),
        float(K_np[1, 2]),
    )


def intrinsics_from_mosca_camera(cams: MonocularCameras, tid: int, Intrinsics):
    K = cams.K(int(cams.default_H), int(cams.default_W)).detach().cpu().numpy()
    return intrinsics_from_K(K, Intrinsics)


def poses_from_T_wc(T_wc_list):
    poses = []
    for T_wc in T_wc_list:
        T_wc = np.asarray(T_wc, dtype=np.float64)
        poses.append(
            Pose(
                T_wc,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
                pose_type=PoseType.CAM_2_WORLD,
            )
        )
    return poses


def load_static_points(logdir: str, device: torch.device):
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(logdir, f"photometric_s_model_{GS_BACKEND.lower()}.pth"),
            map_location="cpu",
        ),
        device=device,
    ).to(device)
    xyz = s_model.get_x.detach().cpu().numpy()
    rgb = SH2RGB(s_model._features_dc.detach().cpu().numpy())
    rgb = np.clip(rgb, 0.0, 1.0)
    return xyz, rgb


def load_dynamic_points(logdir: str, device: torch.device, tid: int):
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(logdir, f"photometric_d_model_{GS_BACKEND.lower()}.pth"),
            map_location="cpu",
        ),
        device=device,
    ).to(device)
    d_model.eval()
    xyz, _, _, _, _ = d_model(tid)
    rgb = SH2RGB(d_model._features_dc.detach().cpu().numpy())
    rgb = np.clip(rgb, 0.0, 1.0)
    return xyz.detach().cpu().numpy(), rgb


def get_training_reference(cfg, ws, optimized_T_wc):
    if cfg is None:
        return None

    if getattr(cfg, "known_camera_pose_path", None) is not None and getattr(
        cfg, "known_camera_intrinsics_path", None
    ) is not None:
        priors = load_vipe_camera_priors(
            pose_npz_path=getattr(cfg, "known_camera_pose_path"),
            intrinsics_npz_path=getattr(cfg, "known_camera_intrinsics_path"),
            expected_T=len(optimized_T_wc),
            camera_convention=getattr(cfg, "known_camera_convention", "opencv"),
        )
        return {
            "T_wc": priors["T_wc"].detach().cpu().numpy(),
            "K": priors["K"].detach().cpu().numpy(),
            "name": "input_train",
        }

    if ws is None:
        return None

    mode = getattr(cfg, "mode", "wild")
    if mode == "iphone":
        (
            gt_training_cam_T_wi,
            _,
            _,
            _,
            _gt_training_fov,
            _,
            gt_training_cxcy_ratio,
            _,
        ) = load_iphone_gt_poses(ws, getattr(cfg, "t_subsample", 1))
        H = int(optimized_T_wc.shape[0] > 0)  # dummy, K comes from solved cams elsewhere
        return {
            "T_wc": gt_training_cam_T_wi.detach().cpu().numpy(),
            "cxcy_ratio": np.asarray(gt_training_cxcy_ratio[0], dtype=np.float32),
            "fovdeg": None,
            "name": "input_train",
        }
    if mode == "nvidia":
        gt_training_cam_T_wi, gt_training_fov, gt_training_cxcy_ratio = load_nvidia_gt_pose(
            osp.join(ws, "poses_bounds.npy")
        )
        return {
            "T_wc": gt_training_cam_T_wi.detach().cpu().numpy(),
            "cxcy_ratio": np.asarray(gt_training_cxcy_ratio[0], dtype=np.float32),
            "fovdeg": float(gt_training_fov),
            "name": "input_train",
        }
    return None


def align_sequence_to_optimized(reference_T_wc, optimized_T_wc, sequence_T_wc):
    aligned = align_ate_c2b_use_a2b(
        torch.as_tensor(reference_T_wc).float(),
        torch.as_tensor(optimized_T_wc).float(),
        torch.as_tensor(sequence_T_wc).float(),
    )
    return aligned.detach().cpu().numpy()


def get_input_training_sequence(
    cfg, ws, optimized_cams, Intrinsics, align_to_optimized=False
):
    optimized_T_wc = optimized_cams.T_wc_list().detach().cpu().numpy()
    reference = get_training_reference(cfg, ws, optimized_T_wc)
    if reference is None:
        return None

    if align_to_optimized:
        T_wc = align_sequence_to_optimized(
            reference["T_wc"], optimized_T_wc, reference["T_wc"]
        )
    else:
        T_wc = reference["T_wc"]

    if "K" in reference:
        intrinsics = [intrinsics_from_K(reference["K"], Intrinsics)] * len(T_wc)
    else:
        intrinsics = [intrinsics_from_mosca_camera(optimized_cams, 0, Intrinsics)] * len(
            T_wc
        )

    return {
        "name": "input_train",
        "T_wc": T_wc,
        "intrinsics": intrinsics,
    }


def get_explicit_test_sequence(args, cfg, optimized_cams, Intrinsics):
    if args.test_camera_pose_path is None or args.test_camera_intrinsics_path is None:
        return None

    optimized_T_wc = optimized_cams.T_wc_list().detach().cpu().numpy()
    reference = get_training_reference(cfg, args.ws, optimized_T_wc)
    train_reference_T_wc = optimized_T_wc if reference is None else reference["T_wc"]

    test_priors = load_vipe_camera_priors(
        pose_npz_path=args.test_camera_pose_path,
        intrinsics_npz_path=args.test_camera_intrinsics_path,
        expected_T=None,
        camera_convention=args.test_camera_convention,
    )
    aligned = align_sequence_to_optimized(
        train_reference_T_wc,
        optimized_T_wc,
        test_priors["T_wc"].detach().cpu().numpy(),
    )
    intrinsics = [intrinsics_from_K(test_priors["K"].detach().cpu().numpy(), Intrinsics)] * len(
        aligned
    )
    return {
        "name": "test_npz",
        "T_wc": aligned,
        "intrinsics": intrinsics,
        "model_tids": test_priors["inds"].detach().cpu().numpy().tolist(),
    }


def add_point_cloud(plotter, points, colors, point_size, opacity=1.0):
    import pyvista as pv

    cloud = pv.PolyData(points)
    cloud["colors"] = colors
    plotter.add_mesh(
        cloud,
        scalars="colors",
        rgb=True,
        point_size=point_size,
        render_points_as_spheres=True,
        opacity=opacity,
    )


def add_camera_set(
    plotter,
    name,
    T_wc_list,
    intrinsics_list,
    color,
    stride,
    camera_scale,
    add_camera_frustum,
):
    poses = poses_from_T_wc(T_wc_list)
    for idx in range(0, len(poses), stride):
        add_camera_frustum(
            p=plotter,
            pose=poses[idx],
            intrinsics=intrinsics_list[idx],
            color=color,
            size=camera_scale,
        )


def build_intrinsics_list_for_camera_set(camera_set):
    if "intrinsics" in camera_set:
        return camera_set["intrinsics"]
    if "K" in camera_set:
        return [intrinsics_from_K(camera_set["K"], Intrinsics)] * len(camera_set["T_wc"])
    raise ValueError(
        f"Camera set '{camera_set.get('name', '<unnamed>')}' must provide either 'intrinsics' or 'K'"
    )


def visualize_camera_sets(
    camera_sets,
    point_sets=None,
    stride=1,
    camera_scale=0.15,
    screenshot=None,
    title_lines=None,
):
    plotter = pv.Plotter(off_screen=screenshot is not None)
    add_coordinate_axes(plotter)

    if title_lines is None:
        title_lines = [
            "MoSca Cameras",
            "OpenCV camera convention; saved poses are T_wc (cam->world)",
        ]
    legend_lines = list(title_lines)
    for camera_set in camera_sets:
        legend_lines.append(f"{camera_set['color']}: {camera_set['name']}")
    plotter.add_text(
        "\n".join(legend_lines),
        position="upper_left",
        font_size=12,
        color="black",
    )

    if point_sets is not None:
        for point_set in point_sets:
            add_point_cloud(
                plotter,
                point_set["points"],
                point_set["colors"],
                point_size=point_set.get("point_size", 3.0),
                opacity=point_set.get("opacity", 1.0),
            )

    for camera_set in camera_sets:
        add_camera_set(
            plotter,
            name=camera_set["name"],
            T_wc_list=camera_set["T_wc"],
            intrinsics_list=build_intrinsics_list_for_camera_set(camera_set),
            color=camera_set["color"],
            stride=stride,
            camera_scale=camera_scale,
            add_camera_frustum=add_camera_frustum,
        )

    plotter.show_axes()
    plotter.view_vector([0, 0, -1], viewup=[0, -1, 0])
    if screenshot is not None:
        plotter.show(screenshot=screenshot)
        logging.info("Saved screenshot to %s", screenshot)
    else:
        plotter.show()
    return


def main():
    args = parse_args()
    configure_logging()

    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")

    if not (
        args.show_input_cameras
        or args.show_optimized_cameras
        or args.show_bundle_cameras
        or args.show_test_cameras
        or args.show_static_points
        or args.show_dynamic_points
    ):
        args.show_input_cameras = True
        args.show_optimized_cameras = True

    cfg = load_cfg(args.cfg)
    device = torch.device(args.device)
    optimized_cams = load_optimized_cameras(args.logdir, device=torch.device("cpu"))
    plotter = pv.Plotter(off_screen=args.screenshot is not None)
    add_coordinate_axes(plotter)
    legend_lines = [
        "MoSca Cameras",
        "OpenCV camera convention; saved poses are T_wc (cam->world)",
        "yellow: input training cameras",
        "orange: bundle cameras",
        "cyan: optimized cameras",
        "red: test cameras",
    ]
    plotter.add_text(
        "\n".join(legend_lines),
        position="upper_left",
        font_size=12,
        color="black",
    )

    if args.show_static_points:
        xyz, rgb = load_static_points(args.logdir, device=device)
        add_point_cloud(plotter, xyz, rgb, point_size=args.point_size, opacity=0.8)

    if args.show_dynamic_points:
        tid = int(np.clip(args.dynamic_time, 0, optimized_cams.T - 1))
        xyz, rgb = load_dynamic_points(args.logdir, device=device, tid=tid)
        add_point_cloud(plotter, xyz, rgb, point_size=args.point_size, opacity=0.5)

    if args.show_input_cameras:
        seq = get_input_training_sequence(
            cfg,
            args.ws,
            optimized_cams,
            Intrinsics,
            align_to_optimized=args.align_input_cameras_to_optimized,
        )
        if seq is not None:
            add_camera_set(
                plotter,
                name=seq["name"],
                T_wc_list=seq["T_wc"],
                intrinsics_list=seq["intrinsics"],
                color="yellow",
                stride=args.stride,
                camera_scale=args.camera_scale,
                add_camera_frustum=add_camera_frustum,
            )
        else:
            logging.warning("Could not determine input training cameras from cfg/ws")

    if args.show_bundle_cameras:
        bundle_cams = load_bundle_cameras(args.logdir, device=torch.device("cpu"))
        if bundle_cams is not None:
            T_wc = bundle_cams.T_wc_list().detach().cpu().numpy()
            intrinsics = [
                intrinsics_from_mosca_camera(bundle_cams, i, Intrinsics)
                for i in range(bundle_cams.T)
            ]
            add_camera_set(
                plotter,
                name="bundle",
                T_wc_list=T_wc,
                intrinsics_list=intrinsics,
                color="orange",
                stride=args.stride,
                camera_scale=args.camera_scale,
                add_camera_frustum=add_camera_frustum,
            )
        else:
            logging.warning("bundle/bundle_cams.pth not found under %s", args.logdir)

    if args.show_optimized_cameras:
        T_wc = optimized_cams.T_wc_list().detach().cpu().numpy()
        intrinsics = [
            intrinsics_from_mosca_camera(optimized_cams, i, Intrinsics)
            for i in range(optimized_cams.T)
        ]
        add_camera_set(
            plotter,
            name="optimized",
            T_wc_list=T_wc,
            intrinsics_list=intrinsics,
            color="cyan",
            stride=args.stride,
            camera_scale=args.camera_scale,
            add_camera_frustum=add_camera_frustum,
        )

    if args.show_test_cameras:
        explicit_seq = get_explicit_test_sequence(args, cfg, optimized_cams, Intrinsics)
        sequences = [explicit_seq] if explicit_seq is not None else []
        if len(sequences) == 0:
            logging.warning("No test-camera sequences found")
        for seq in sequences:
            add_camera_set(
                plotter,
                name=seq["name"],
                T_wc_list=seq["T_wc"],
                intrinsics_list=seq["intrinsics"],
                color="red",
                stride=args.stride,
                camera_scale=args.camera_scale,
                add_camera_frustum=add_camera_frustum,
            )

    plotter.show_axes()
    plotter.view_vector([0, 0, -1], viewup=[0, -1, 0])
    if args.screenshot is not None:
        plotter.show(screenshot=args.screenshot)
        logging.info("Saved screenshot to %s", args.screenshot)
    else:
        plotter.show()


if __name__ == "__main__":
    main()
