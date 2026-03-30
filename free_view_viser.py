"""
Interactive free-viewpoint rendering of MoSca 4D scenes using Viser.

Usage:
    python free_view_viser.py \
        --cfg ./profile/demo/demo_fit.yaml \
        --logdir ./demo/duck/logs/demo_fit_native_add3_20260317_202146 \
        [--port 8080] [--host 0.0.0.0] [--render_scale 0.5]
"""

import os, sys, os.path as osp
import argparse
import time
import numpy as np
import torch
import viser
import viser.transforms as vtf

sys.path.append(osp.dirname(osp.abspath(__file__)))

from omegaconf import OmegaConf
from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian
from lib_render.render_helper import render, GS_BACKEND


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

@torch.no_grad()
def load_scene(cfg_path: str, logdir: str, device: torch.device):
    cfg = OmegaConf.load(cfg_path)

    cams: MonocularCameras = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(logdir, "photometric_cam.pth"), map_location=device)
    )

    s_model: StaticGaussian = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(logdir, f"photometric_s_model_{GS_BACKEND.lower()}.pth"),
            map_location=device,
        ),
        device=device,
    )

    d_model: DynSCFGaussian = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(logdir, f"photometric_d_model_{GS_BACKEND.lower()}.pth"),
            map_location=device,
        ),
        device=device,
    )

    # Ensure ALL parameters are on the correct device
    cams = cams.to(device)
    s_model = s_model.to(device)
    d_model = d_model.to(device)
    cams.eval()
    s_model.eval()
    d_model.eval()

    # Bake skinning for faster forward pass
    d_model.set_inference_mode()

    return cfg, cams, s_model, d_model


# ---------------------------------------------------------------------------
#  Rendering helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_frame(
    s_model,
    d_model,
    T_cw: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    t: int,
    bg_color=(1.0, 1.0, 1.0),
    show_static: bool = True,
    show_dynamic: bool = True,
):
    """Render a single frame and return an (H, W, 3) uint8 numpy array."""
    gs5 = []
    if show_static:
        gs5.append(s_model())
    if show_dynamic:
        gs5.append(d_model(t))
    if len(gs5) == 0:
        return np.ones((H, W, 3), dtype=np.uint8) * 255

    render_dict = render(
        gs5, H, W,
        K=K,
        T_cw=T_cw,
        bg_color=list(bg_color),
    )
    rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
    return (rgb.detach().cpu().numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
#  Coordinate conversion:  MoSca (OpenCV) <-> Viser (OpenGL)
#
#  OpenCV:  +X right, +Y down,  +Z forward
#  OpenGL:  +X right, +Y up,    -Z forward
#  Flip = diag(1, -1, -1)   applied on the camera-local axes.
# ---------------------------------------------------------------------------

_FLIP = np.diag([1.0, -1.0, -1.0])


def viser_cam_to_T_cw(camera: viser.CameraHandle) -> np.ndarray:
    """Viser camera → 4×4 T_cw (OpenCV)."""
    R_wc_gl = vtf.SO3(camera.wxyz).as_matrix()
    t_wc = np.array(camera.position)
    R_wc_cv = R_wc_gl @ _FLIP
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc_cv
    T_wc[:3, 3] = t_wc
    return np.linalg.inv(T_wc)


def T_cw_to_viser_pose(T_cw_np: np.ndarray):
    """4×4 T_cw (OpenCV) → (wxyz, position) for viser."""
    T_wc = np.linalg.inv(T_cw_np)
    R_wc_cv = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    R_wc_gl = R_wc_cv @ _FLIP
    wxyz = vtf.SO3.from_matrix(R_wc_gl).wxyz
    return wxyz, t_wc


def get_fov_y(cams, H, W):
    """Compute vertical FOV (radians) from MoSca rel_focal."""
    # rel_focal = f_pixel / (L/2),  L = min(H, W)
    rel_fy = float(cams.rel_focal[1].detach().cpu())
    L = min(H, W)
    fy_pixel = rel_fy * L / 2.0
    fov_y = 2.0 * np.arctan(H / (2.0 * fy_pixel))
    return fov_y


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("MoSca Viser Free-View Renderer")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--render_scale", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print(f"Loading scene from {args.logdir} ...")
    cfg, cams, s_model, d_model = load_scene(args.cfg, args.logdir, device)

    T_total = cams.T
    H_orig = int(cams.default_H)
    W_orig = int(cams.default_W)
    fov_y_default = get_fov_y(cams, H_orig, W_orig)
    print(f"Scene loaded: {T_total} frames, {H_orig}x{W_orig}, "
          f"fov_y={np.rad2deg(fov_y_default):.1f} deg, GS_BACKEND={GS_BACKEND}")

    # Pre-compute all training camera poses (numpy, on CPU)
    training_T_cw = []
    for t_idx in range(T_total):
        training_T_cw.append(cams.T_cw(t_idx).detach().cpu().numpy())

    # ------------------------------------------------------------------
    #  Viser server
    # ------------------------------------------------------------------
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Viser server started at http://{args.host}:{args.port}")

    # ---- GUI: Playback ----
    with server.gui.add_folder("Playback"):
        gui_playing = server.gui.add_checkbox("Play", initial_value=False)
        gui_timestep = server.gui.add_slider(
            "Timestep", min=0, max=T_total - 1, step=1, initial_value=0,
        )
        gui_fps = server.gui.add_slider(
            "FPS", min=1.0, max=60.0, step=1.0, initial_value=args.fps,
        )
        gui_loop = server.gui.add_checkbox("Loop", initial_value=True)

    # ---- GUI: Rendering ----
    with server.gui.add_folder("Rendering"):
        gui_show_static = server.gui.add_checkbox("Static GS", initial_value=True)
        gui_show_dynamic = server.gui.add_checkbox("Dynamic GS", initial_value=True)
        gui_bg_white = server.gui.add_checkbox("White Background", initial_value=True)
        gui_render_scale = server.gui.add_slider(
            "Render Scale", min=0.1, max=1.0, step=0.05,
            initial_value=args.render_scale,
        )

    # ---- GUI: Camera ----
    with server.gui.add_folder("Camera"):
        gui_follow_training = server.gui.add_checkbox(
            "Follow Training View", initial_value=False,
        )
        gui_snap_btn = server.gui.add_button("Snap to Current Timestep")
        gui_cam_info = server.gui.add_text("Info", initial_value="", disabled=True)

    # ------------------------------------------------------------------
    #  Snap to training view (one-shot)
    # ------------------------------------------------------------------
    @gui_snap_btn.on_click
    def _snap(event: viser.GuiEvent) -> None:
        if event.client is None:
            return
        t = int(gui_timestep.value)
        wxyz, position = T_cw_to_viser_pose(training_T_cw[t])
        event.client.camera.wxyz = wxyz
        event.client.camera.position = position

    # ------------------------------------------------------------------
    #  Initial camera for new clients + space-bar JS injection
    # ------------------------------------------------------------------

    # We inject a small JS snippet so pressing Space toggles Play.
    # This is added once on the server (shared HTML), not per-client.
    _space_js = server.gui.add_html(
        """
        <script>
        (function() {
            if (window.__mosca_kb) return;
            window.__mosca_kb = true;
            document.addEventListener('keydown', function(e) {
                if (e.code === 'Space' && e.target === document.body) {
                    e.preventDefault();
                    // Toggle the first checkbox whose label contains "Play"
                    var labels = document.querySelectorAll('label');
                    for (var i = 0; i < labels.length; i++) {
                        if (labels[i].textContent.indexOf('Play') !== -1) {
                            var inp = labels[i].querySelector('input[type="checkbox"]');
                            if (inp) { inp.click(); break; }
                        }
                    }
                }
            });
        })();
        </script>
        """
    )

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle) -> None:
        wxyz, position = T_cw_to_viser_pose(training_T_cw[0])
        client.camera.wxyz = wxyz
        client.camera.position = position
        client.camera.fov = fov_y_default

    # ------------------------------------------------------------------
    #  Playback logic
    # ------------------------------------------------------------------
    last_advance_time = time.time()

    def playback_tick():
        nonlocal last_advance_time
        if not gui_playing.value:
            last_advance_time = time.time()
            return
        now = time.time()
        interval = 1.0 / max(float(gui_fps.value), 1.0)
        if now - last_advance_time >= interval:
            last_advance_time = now
            t = int(gui_timestep.value) + 1
            if t >= T_total:
                if gui_loop.value:
                    t = 0
                else:
                    t = T_total - 1
                    gui_playing.value = False
            gui_timestep.value = t

    # ------------------------------------------------------------------
    #  Main render loop
    # ------------------------------------------------------------------
    print("Ready. Open the URL in your browser.  Press Space to play/pause.")
    while True:
        playback_tick()

        clients = server.get_clients()
        if len(clients) == 0:
            time.sleep(0.05)
            continue

        t = int(gui_timestep.value)
        t = min(max(t, 0), T_total - 1)

        for client in clients.values():
            cam = client.camera

            # --- Follow training view: continuously move camera ---
            if gui_follow_training.value:
                wxyz, position = T_cw_to_viser_pose(training_T_cw[t])
                cam.wxyz = wxyz
                cam.position = position

            # --- Build T_cw from (possibly overridden) viser camera ---
            T_cw_np = viser_cam_to_T_cw(cam)
            T_cw_torch = torch.from_numpy(T_cw_np).float().to(device)

            # --- Build intrinsics from client viewport FOV ---
            cur_scale = float(gui_render_scale.value)
            H_r = max(1, int(H_orig * cur_scale))
            W_r = max(1, int(W_orig * cur_scale))
            fov_y = float(cam.fov)
            fy = H_r / (2.0 * np.tan(fov_y / 2.0))
            fx = fy
            cx, cy = W_r / 2.0, H_r / 2.0
            K = torch.tensor(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                dtype=torch.float32, device=device,
            )

            bg = (1.0, 1.0, 1.0) if gui_bg_white.value else (0.0, 0.0, 0.0)

            img = render_frame(
                s_model, d_model,
                T_cw=T_cw_torch, K=K,
                H=H_r, W=W_r, t=t,
                bg_color=bg,
                show_static=gui_show_static.value,
                show_dynamic=gui_show_dynamic.value,
            )

            client.scene.set_background_image(img, format="jpeg")

            gui_cam_info.value = (
                f"t={t}/{T_total-1}  {W_r}x{H_r}  "
                f"fov={np.rad2deg(fov_y):.1f}"
                + ("  [following]" if gui_follow_training.value else "")
            )

        time.sleep(0.005)


if __name__ == "__main__":
    main()
