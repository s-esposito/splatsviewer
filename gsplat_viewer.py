"""Standalone real-time Gaussian Splatting viewer using gsplat + DearPyGui.

Usage:
    python gsplat_viewer.py [ply_path] [--cameras <json_path>] [--width W] [--height H]

Arguments:
    ply_path               Path to a .ply file (optional, can load from GUI)
    --cameras <json_path>  Optional JSON with context/target cameras
    --width  W             Render width  (default: 1920)
    --height H             Render height (default: 1080)

Camera JSON format:
    {
        "scene": "<id>",
        "context": {"extrinsics": [4x4, ...], "intrinsics": [3x3, ...]},
        "target":  {"extrinsics": [4x4, ...], "intrinsics": [3x3, ...]},
        "resolution": [H, W]
    }
    Extrinsics are c2w matrices. Intrinsics are normalized (fx<1, cx~0.5).

Controls (orbit mode):
    Left-click drag    : orbit
    Right-click drag   : pan
    Scroll wheel       : zoom
    W/A/S/D            : move camera
    Shift + move       : fast move
    R                  : reset camera
    F12                : save screenshot to screenshot.png

GUI panel:
    shader             : RGB | Depth (expected) | Depth (accumulated) | Weights | Alphas
    view               : Orbit (interactive) or any loaded camera
    radius_clip, eps2d, sh_degree, rasterize_mode, near/far plane
"""

import argparse
import json
import math
import os
import time
import numpy as np
import cv2
import torch
import dearpygui.dearpygui as dpg
from plyfile import PlyData
from gsplat.rendering import rasterization
from orbit_camera import OrbitCamera

# SH basis constant for degree 0
SH_C0 = 0.28209479177387814

# Widget tags
TAG_RENDER_TEX = "render_tex"
TAG_RENDER_IMG = "render_img"
TAG_RADIUS_CLIP = "slider_radius_clip"
TAG_EPS2D = "slider_eps2d"
TAG_SH_DEGREE = "combo_sh_degree"
TAG_RASTERIZE_MODE = "combo_rasterize_mode"
TAG_NEAR_PLANE = "slider_near_plane"
TAG_FAR_PLANE = "slider_far_plane"
TAG_SHADER = "combo_shader"
TAG_CAMERA_SELECT = "combo_camera"
TAG_SCALE_CLIP_MIN_ENABLE = "checkbox_scale_clip_min"
TAG_SCALE_CLIP_MIN_VALUE = "slider_scale_clip_min"
TAG_SCALE_CLIP_MAX_ENABLE = "checkbox_scale_clip_max"
TAG_SCALE_CLIP_MAX_VALUE = "slider_scale_clip_max"
TAG_WEIGHT_REF_CAM = "combo_weight_ref_cam"
TAG_WEIGHT_FILTER_ENABLE = "checkbox_weight_filter"
TAG_WEIGHT_FILTER_MODE = "combo_weight_filter_mode"
TAG_WEIGHT_FILTER_THRESH = "slider_weight_filter_thresh"
TAG_WEIGHT_FILTER_GROUP = "group_weight_filter"
TAG_INFO_TEXT = "info_text"
TAG_MAIN_WIN = "main_win"
TAG_CANVAS_WIN = "canvas_win"
TAG_FILE_DIALOG_PLY = "file_dialog_ply"
TAG_FILE_DIALOG_CAMERAS = "file_dialog_cameras"
TAG_FILE_DIALOG_SAVE = "file_dialog_save"
TAG_BG_COLOR = "color_bg"


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------

def load_ply(path: str, device: str = "cuda") -> dict:
    """Load a standard 3DGS .ply file and return Gaussian attributes as tensors.

    Supports:
      - Original 3DGS format (f_dc_*, f_rest_*, scale_*, rot_*, opacity)
      - Vertex-color format (red/green/blue 0-255)
      - Custom feature format (feature_0/1/2)
    """
    print(f"Loading PLY from {path} ...")
    plydata = PlyData.read(path)
    vtx = plydata["vertex"]
    prop_names = {p.name for p in vtx.properties}
    n = vtx.count
    print(f"  {n:,} Gaussians")

    # --- positions ---
    means = np.stack([vtx["x"], vtx["y"], vtx["z"]], axis=-1).astype(np.float32)

    # --- SH coefficients and colors ---
    sh_coeffs = None
    max_sh_degree = -1

    if "f_dc_0" in prop_names:
        dc = np.stack([vtx["f_dc_0"], vtx["f_dc_1"], vtx["f_dc_2"]], axis=-1).astype(np.float32)
        rest_names = sorted(
            [p.name for p in vtx.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        n_rest = len(rest_names)

        if n_rest > 0 and n_rest % 3 == 0:
            K_minus_1 = n_rest // 3
            K = K_minus_1 + 1
            max_sh_degree = int(math.sqrt(K)) - 1
            print(f"  SH degree {max_sh_degree} ({K} coefficients per color channel)")
            rest = np.zeros((n, n_rest), dtype=np.float32)
            for i, name in enumerate(rest_names):
                rest[:, i] = np.array(vtx[name], dtype=np.float32)
            rest = rest.reshape(n, 3, K_minus_1).transpose(0, 2, 1)
            sh_coeffs = np.concatenate([dc[:, np.newaxis, :], rest], axis=1)
        else:
            max_sh_degree = 0
            sh_coeffs = dc[:, np.newaxis, :]
            print(f"  SH degree 0 (DC only)")

        # if dc > 1e10 or dc < -1e10:
        #     print(f"  Warning: DC values have large magnitude, clamping for color display")
        dc = np.clip(dc, -10.0, 10.0)
        colors = 1.0 / (1.0 + np.exp(-(SH_C0 * dc + 0.5)))
    elif "red" in prop_names:
        colors = np.stack([vtx["red"], vtx["green"], vtx["blue"]], axis=-1).astype(np.float32) / 255.0
    elif "feature_0" in prop_names:
        colors = np.stack([vtx["feature_0"], vtx["feature_1"], vtx["feature_2"]], axis=-1).astype(np.float32)
    else:
        print("  Warning: no color attributes found, using white")
        colors = np.ones((n, 3), dtype=np.float32)

    # --- scales ---
    if "scale_0" in prop_names:
        scale_names = sorted(
            [p.name for p in vtx.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scale_cols = [np.array(vtx[s], dtype=np.float32) for s in scale_names]
        scales = np.stack(scale_cols, axis=-1)
        scales = np.exp(scales)
        if scales.shape[-1] == 1:
            scales = np.broadcast_to(scales, (n, 3)).copy()
        elif scales.shape[-1] == 2:
            pad = scales.mean(axis=-1, keepdims=True)
            scales = np.concatenate([scales, pad], axis=-1)
    else:
        print("  Warning: no scale attributes, using default")
        scales = np.full((n, 3), 0.01, dtype=np.float32)

    # --- rotations ---
    if "rot_0" in prop_names:
        quats = np.stack([vtx["rot_0"], vtx["rot_1"], vtx["rot_2"], vtx["rot_3"]], axis=-1).astype(np.float32)
        norms = np.linalg.norm(quats, axis=-1, keepdims=True)
        quats = quats / np.clip(norms, 1e-8, None)
    else:
        print("  Warning: no rotation attributes, using identity")
        quats = np.zeros((n, 4), dtype=np.float32)
        quats[:, 0] = 1.0

    # --- opacities ---
    if "opacity" in prop_names:
        opacities = np.array(vtx["opacity"], dtype=np.float32)
        opacities = 1.0 / (1.0 + np.exp(-opacities))
    else:
        print("  Warning: no opacity attribute, using 1.0")
        opacities = np.ones(n, dtype=np.float32)
                
    # filter out Gaussians with nans or infs in any attribute
    valid_mask = np.ones(n, dtype=bool)
    for arr in [means, colors, scales, quats, opacities]:
        # check if any nan or inf values in the last axis (e.g. color channels) and mark the whole Gaussian as invalid if so
        valid_mask &= np.isfinite(arr).all(axis=-1)
    # print a warning if any invalid Gaussians were found and will be ignored
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  Warning: {n_invalid} Gaussians contain NaN or Inf values and will be ignored")
        means = means[valid_mask]
        colors = colors[valid_mask]
        scales = scales[valid_mask]
        quats = quats[valid_mask]
        opacities = opacities[valid_mask]
        if sh_coeffs is not None:
            sh_coeffs = sh_coeffs[valid_mask]

    if len(means) == 0:
        print("  Warning: all Gaussians were filtered out, scene will be empty")
    
    # for each attribute, check if there are nan or inf values and print a warning if so
    for name, arr in [("means", means), ("colors", colors), ("scales", scales),
                        ("quats", quats), ("opacities", opacities)]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"  Warning: {name} contains NaN or Inf values")
    
    result = {
        "means": torch.from_numpy(means).to(device),
        "colors": torch.from_numpy(colors).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "quats": torch.from_numpy(quats).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
        "max_sh_degree": max_sh_degree,
        "sh_coeffs": torch.from_numpy(sh_coeffs).to(device) if sh_coeffs is not None else None,
    }

    bmin = means.min(axis=0) if len(means) > 0 else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    bmax = means.max(axis=0) if len(means) > 0 else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    center = (bmin + bmax) / 2.0
    extent = np.linalg.norm(bmax - bmin)
    print(f"  Bounds: [{bmin}] -> [{bmax}]")
    print(f"  Center: {center}, Extent: {extent:.3f}")
    result["center"] = center
    result["extent"] = float(extent)
    return result


# ---------------------------------------------------------------------------
# Camera JSON loading
# ---------------------------------------------------------------------------

def load_cameras(path: str, render_w: int, render_h: int) -> dict:
    """Load cameras from JSON.

    Expected format:
        scene: str
        context: {extrinsics: list[4x4], intrinsics: list[3x3]}
        target:  {extrinsics: list[4x4], intrinsics: list[3x3]}
        resolution: [H, W]

    Intrinsics are normalized (fx/fy < 1, cx=cy~0.5) and get scaled to render resolution.
    Extrinsics are c2w matrices.

    Returns dict mapping camera name -> {"w2c": (4,4) np, "K": (3,3) np}.
    """
    print(f"Loading cameras from {path} ...")
    data = json.load(open(path))

    cameras = {}
    for group in ("context", "target"):
        if group not in data:
            continue
        prefix = "ctx" if group == "context" else "tgt"
        extrinsics = data[group]["extrinsics"]
        intrinsics = data[group]["intrinsics"]
        for i, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
            c2w = np.array(ext, dtype=np.float32)
            w2c = np.linalg.inv(c2w).astype(np.float32)
            K_norm = np.array(intr, dtype=np.float32)
            # denormalize intrinsics to render resolution
            K = K_norm.copy()
            K[0, :] *= render_w   # fx, skew, cx
            K[1, :] *= render_h   # fy, cy
            name = f"{prefix}_{i}"
            cameras[name] = {"w2c": w2c, "K": K}

    print(f"  Loaded {len(cameras)} cameras ({list(cameras.keys())[0]} .. {list(cameras.keys())[-1]})")
    return cameras


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class GsplatViewer:
    PANEL_WIDTH = 280

    def __init__(self, ply_path: str = None, cameras_path: str = None,
                 width: int = 1280, height: int = 720):
        self.render_w = width
        self.render_h = height
        self.device = "cuda"

        # load gaussians (optional at startup)
        self.gs = None
        if ply_path is not None:
            self.gs = load_ply(ply_path, device=self.device)

        # load cameras from JSON (optional at startup)
        self.loaded_cameras = {}
        if cameras_path is not None:
            self.loaded_cameras = load_cameras(cameras_path, width, height)

        max_sh = self.gs["max_sh_degree"] if self.gs is not None else -1
        default_sh = max_sh if max_sh >= 0 else None

        # render parameters
        self.render_params = {
            "radius_clip": 0.0,
            "eps2d": 0.3,
            "sh_degree": default_sh,
            "rasterize_mode": "antialiased",
            "near_plane": 0.01,
            "far_plane": 1000.0,
        }

        # camera
        if self.gs is not None:
            center = self.gs["center"]
            radius = self.gs["extent"] * 0.7
        else:
            center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            radius = 3.0
        self.camera = OrbitCamera(
            width=width, height=height,
            radius=max(radius, 0.5), fovy=50.0,
            near=0.01, far=1000.0,
            center=center, up="y",
            azimuth_deg=45.0, elevation_deg=30.0,
        )

        # mouse state
        self._last_mouse = np.array([0.0, 0.0])
        self._mouse_in_canvas = False

        # pre-allocate RGBA texture buffer
        self._tex_data = np.zeros(self.render_w * self.render_h * 3, dtype=np.float32)

        # --- build DearPyGui UI ---
        dpg.create_context()

        viewport_w = width + self.PANEL_WIDTH + 20
        viewport_h = height + 40
        dpg.create_viewport(title="gsplat viewer", width=viewport_w, height=viewport_h,
                            resizable=False)

        # texture for rendering
        with dpg.texture_registry():
            dpg.add_raw_texture(
                self.render_w, self.render_h, self._tex_data,
                tag=TAG_RENDER_TEX, format=dpg.mvFormat_Float_rgb,
            )

        # file dialogs
        with dpg.file_dialog(directory_selector=False, show=False,
                             callback=self._on_ply_selected,
                             tag=TAG_FILE_DIALOG_PLY,
                             width=700, height=400):
            dpg.add_file_extension(".ply", color=(0, 255, 0, 255))
            dpg.add_file_extension(".*")

        with dpg.file_dialog(directory_selector=False, show=False,
                             callback=self._on_cameras_selected,
                             tag=TAG_FILE_DIALOG_CAMERAS,
                             width=700, height=400):
            dpg.add_file_extension(".json", color=(0, 255, 0, 255))
            dpg.add_file_extension(".*")

        with dpg.file_dialog(directory_selector=False, show=False,
                             callback=self._on_save_selected,
                             tag=TAG_FILE_DIALOG_SAVE,
                             default_filename="render.png",
                             width=700, height=400):
            dpg.add_file_extension(".png", color=(0, 255, 0, 255))

        # main window
        with dpg.window(tag=TAG_MAIN_WIN, no_title_bar=True, no_close=True,
                        no_move=True, no_resize=True):
            with dpg.group(horizontal=True):
                # left: render canvas
                dpg.add_image(TAG_RENDER_TEX, tag=TAG_RENDER_IMG,
                              width=self.render_w, height=self.render_h)

                # right: controls panel
                with dpg.child_window(width=self.PANEL_WIDTH, height=self.render_h):
                    dpg.add_button(label="Load PLY...", width=-1,
                                   callback=lambda: dpg.show_item(TAG_FILE_DIALOG_PLY))
                    dpg.add_button(label="Load Cameras...", width=-1,
                                   callback=lambda: dpg.show_item(TAG_FILE_DIALOG_CAMERAS))
                    dpg.add_button(label="Save Screenshot...", width=-1,
                                   callback=lambda: dpg.show_item(TAG_FILE_DIALOG_SAVE))

                    dpg.add_separator()
                    dpg.add_text("Render Parameters", color=(200, 200, 255))
                    dpg.add_separator()

                    shader_items = ["RGB", "Depth (expected)", "Depth (accumulated)", "Alphas"]
                    if self.loaded_cameras:
                        shader_items.insert(3, "Weights")
                    dpg.add_combo(
                        label="shader", tag=TAG_SHADER,
                        items=shader_items,
                        default_value="RGB", width=140,
                    )

                    # weight controls (only shown when Weights shader is active)
                    cam_names = list(self.loaded_cameras.keys())
                    with dpg.group(tag=TAG_WEIGHT_FILTER_GROUP, show=False):
                        dpg.add_combo(
                            label="ref camera", tag=TAG_WEIGHT_REF_CAM,
                            items=cam_names,
                            default_value=cam_names[0] if cam_names else "",
                            width=140,
                        )
                        dpg.add_checkbox(
                            label="filter by weight",
                            tag=TAG_WEIGHT_FILTER_ENABLE,
                            default_value=False,
                        )
                        dpg.add_combo(
                            label="keep", tag=TAG_WEIGHT_FILTER_MODE,
                            items=["High", "Low"],
                            default_value="High", width=140,
                        )
                        dpg.add_slider_float(
                            label="threshold",
                            tag=TAG_WEIGHT_FILTER_THRESH,
                            default_value=0.0,
                            min_value=0.0, max_value=1.0,
                            width=140, format="%.4f",
                        )

                    dpg.add_slider_float(
                        label="radius_clip", tag=TAG_RADIUS_CLIP,
                        default_value=self.render_params["radius_clip"],
                        min_value=0.0, max_value=100.0, width=140,
                    )
                    dpg.add_slider_float(
                        label="eps2d", tag=TAG_EPS2D,
                        default_value=self.render_params["eps2d"],
                        min_value=0.0, max_value=2.0, width=140,
                        format="%.3f",
                    )

                    if max_sh >= 0:
                        sh_items = ["None"] + [str(i) for i in range(max_sh + 1)]
                        sh_default = str(default_sh) if default_sh is not None else "None"
                    else:
                        sh_items = ["None"]
                        sh_default = "None"
                    dpg.add_combo(
                        label="sh_degree", tag=TAG_SH_DEGREE,
                        items=sh_items, default_value=sh_default, width=140,
                    )
                    dpg.add_combo(
                        label="rasterize_mode", tag=TAG_RASTERIZE_MODE,
                        items=["classic", "antialiased"],
                        default_value=self.render_params["rasterize_mode"], width=140,
                    )
                    dpg.add_color_edit(
                        label="background", tag=TAG_BG_COLOR,
                        default_value=(0, 0, 0, 255),
                        no_alpha=True, width=140,
                    )

                    dpg.add_separator()
                    dpg.add_slider_float(
                        label="near (log10)", tag=TAG_NEAR_PLANE,
                        default_value=math.log10(max(self.render_params["near_plane"], 1e-6)),
                        min_value=-3.0, max_value=2.0, width=140, format="%.2f",
                    )
                    dpg.add_slider_float(
                        label="far (log10)", tag=TAG_FAR_PLANE,
                        default_value=math.log10(max(self.render_params["far_plane"], 1.0)),
                        min_value=1.0, max_value=6.0, width=140, format="%.2f",
                    )

                    dpg.add_separator()
                    dpg.add_checkbox(
                        label="min scale clip",
                        tag=TAG_SCALE_CLIP_MIN_ENABLE,
                        default_value=False,
                    )
                    dpg.add_slider_float(
                        label="min scale", tag=TAG_SCALE_CLIP_MIN_VALUE,
                        default_value=0.005,
                        min_value=0.0, max_value=0.1,
                        width=140, format="%.4f",
                    )
                    dpg.add_checkbox(
                        label="max scale clip",
                        tag=TAG_SCALE_CLIP_MAX_ENABLE,
                        default_value=False,
                    )
                    dpg.add_slider_float(
                        label="max scale", tag=TAG_SCALE_CLIP_MAX_VALUE,
                        default_value=0.05,
                        min_value=0.001, max_value=1.0,
                        width=140, format="%.4f",
                    )

                    dpg.add_separator()
                    dpg.add_text("", tag=TAG_INFO_TEXT, color=(160, 160, 160))

                    dpg.add_separator()
                    dpg.add_text("Camera", color=(200, 200, 255))

                    cam_items = ["Orbit"] + list(self.loaded_cameras.keys())
                    dpg.add_combo(
                        label="view", tag=TAG_CAMERA_SELECT,
                        items=cam_items, default_value="Orbit", width=140,
                    )

                    dpg.add_text("LMB drag: orbit", color=(140, 140, 140))
                    dpg.add_text("RMB drag: pan", color=(140, 140, 140))
                    dpg.add_text("Scroll: zoom", color=(140, 140, 140))
                    dpg.add_text("WASD: move", color=(140, 140, 140))
                    dpg.add_text("R: reset camera", color=(140, 140, 140))
                    dpg.add_text("F12: quick screenshot", color=(140, 140, 140))

        dpg.set_primary_window(TAG_MAIN_WIN, True)

        # mouse / keyboard handlers
        with dpg.handler_registry():
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
            dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
            dpg.add_mouse_click_handler(callback=self._on_mouse_click)
            dpg.add_mouse_release_handler(callback=self._on_mouse_release)

        dpg.setup_dearpygui()
        dpg.show_viewport()

        print(f"Viewer initialized: {width}x{height}")
        if self.gs is not None:
            print(f"  SH degree: {default_sh} (max available: {max_sh})")
        else:
            print(f"  No PLY loaded (use 'Load PLY...' button)")

    # ---------- input handling ----------

    def _is_over_canvas(self) -> bool:
        """Check if mouse is over the render image."""
        return dpg.is_item_hovered(TAG_RENDER_IMG)

    def _using_orbit(self) -> bool:
        return dpg.get_value(TAG_CAMERA_SELECT) == "Orbit"

    def _on_mouse_move(self, sender, app_data):
        mx, my = dpg.get_mouse_pos()
        mouse = np.array([mx, my], dtype=np.float64)
        dx, dy = mouse - self._last_mouse
        self._last_mouse = mouse

        if not self._is_over_canvas() or not self._using_orbit():
            return

        left = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        right = dpg.is_mouse_button_down(dpg.mvMouseButton_Right)
        middle = dpg.is_mouse_button_down(dpg.mvMouseButton_Middle)

        if left:
            self.camera.orbit(dx, dy)
        elif right:
            self.camera.pan(dx, dy)
        elif middle:
            self.camera.pan(dx, dy)

    def _on_mouse_wheel(self, sender, app_data):
        if self._is_over_canvas() and self._using_orbit():
            self.camera.zoom(app_data)

    def _on_mouse_click(self, sender, app_data):
        self._last_mouse = np.array(dpg.get_mouse_pos(), dtype=np.float64)

    def _on_mouse_release(self, sender, app_data):
        pass

    def _handle_keyboard(self):
        """Poll keyboard state each frame for camera movement."""
        if dpg.is_key_pressed(dpg.mvKey_F12):
            self._save_screenshot("screenshot.png")
        if not self._using_orbit():
            return
        shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
        if dpg.is_key_down(dpg.mvKey_W):
            self.camera.move_forward(fast=shift)
        if dpg.is_key_down(dpg.mvKey_S):
            self.camera.move_backward(fast=shift)
        if dpg.is_key_down(dpg.mvKey_A):
            self.camera.move_left(fast=shift)
        if dpg.is_key_down(dpg.mvKey_D):
            self.camera.move_right(fast=shift)
        if dpg.is_key_pressed(dpg.mvKey_R):
            center = self.gs["center"]
            radius = self.gs["extent"] * 0.7
            self.camera.set_center(center)
            self.camera.set_radius(max(radius, 0.5))
            self.camera.set_azimuth_deg(45.0)
            self.camera.set_elevation_deg(30.0)

    # ---------- file loading ----------

    def _on_ply_selected(self, sender, app_data):
        """Callback when a PLY file is selected from the file dialog."""
        path = app_data.get("file_path_name", "")
        if not path:
            return
        try:
            self.gs = load_ply(path, device=self.device)
        except Exception as e:
            print(f"Error loading PLY: {e}")
            return

        # reset orbit camera to new scene
        center = self.gs["center"]
        radius = self.gs["extent"] * 0.7
        self.camera.set_center(center)
        self.camera.set_radius(max(radius, 0.5))
        self.camera.set_azimuth_deg(45.0)
        self.camera.set_elevation_deg(30.0)

        # update SH combo
        max_sh = self.gs["max_sh_degree"]
        if max_sh >= 0:
            sh_items = ["None"] + [str(i) for i in range(max_sh + 1)]
            sh_default = str(max_sh)
        else:
            sh_items = ["None"]
            sh_default = "None"
        dpg.configure_item(TAG_SH_DEGREE, items=sh_items, default_value=sh_default)
        self.render_params["sh_degree"] = max_sh if max_sh >= 0 else None

        # update shader combo (Weights needs cameras)
        self._update_shader_items()

    def _on_cameras_selected(self, sender, app_data):
        """Callback when a cameras JSON file is selected from the file dialog."""
        path = app_data.get("file_path_name", "")
        if not path:
            return
        try:
            self.loaded_cameras = load_cameras(path, self.render_w, self.render_h)
        except Exception as e:
            print(f"Error loading cameras: {e}")
            return

        # update camera select combo
        cam_items = ["Orbit"] + list(self.loaded_cameras.keys())
        dpg.configure_item(TAG_CAMERA_SELECT, items=cam_items, default_value="Orbit")

        # update weight ref camera combo
        cam_names = list(self.loaded_cameras.keys())
        dpg.configure_item(TAG_WEIGHT_REF_CAM,
                           items=cam_names,
                           default_value=cam_names[0] if cam_names else "")

        # update shader combo (add Weights if cameras loaded)
        self._update_shader_items()

    def _on_save_selected(self, sender, app_data):
        """Callback when a save path is selected from the file dialog."""
        path = app_data.get("file_path_name", "")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self._save_screenshot(path)

    def _save_screenshot(self, path: str):
        """Save the current render buffer to a PNG file."""
        image = self._tex_data.reshape(self.render_h, self.render_w, 3)
        image_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image_bgr)
        print(f"Screenshot saved to {path}")

    def _update_shader_items(self):
        """Refresh shader combo items based on current state."""
        items = ["RGB", "Depth (expected)", "Depth (accumulated)", "Alphas"]
        if self.loaded_cameras:
            items.insert(3, "Weights")
        current = dpg.get_value(TAG_SHADER)
        if current not in items:
            current = "RGB"
        dpg.configure_item(TAG_SHADER, items=items, default_value=current)

    # ---------- GUI sync ----------

    def _sync_params_from_gui(self):
        """Read GUI widget values into render_params."""
        self.render_params["shader"] = dpg.get_value(TAG_SHADER)

        # show/hide weight filter controls (available for all shaders when cameras loaded)
        dpg.configure_item(TAG_WEIGHT_FILTER_GROUP,
                           show=bool(self.loaded_cameras))

        self.render_params["radius_clip"] = dpg.get_value(TAG_RADIUS_CLIP)
        self.render_params["eps2d"] = dpg.get_value(TAG_EPS2D)

        sh = dpg.get_value(TAG_SH_DEGREE)
        self.render_params["sh_degree"] = None if sh == "None" else int(sh)

        self.render_params["rasterize_mode"] = dpg.get_value(TAG_RASTERIZE_MODE)
        self.render_params["near_plane"] = round(10 ** dpg.get_value(TAG_NEAR_PLANE), 6)
        self.render_params["far_plane"] = round(10 ** dpg.get_value(TAG_FAR_PLANE), 1)

    # ---------- rendering ----------

    def _get_background(self):
        """Return the background color as a (3,) tensor from the GUI color picker."""
        rgba = dpg.get_value(TAG_BG_COLOR)  # [R, G, B, A] ints 0-255
        return torch.tensor([rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0],
                            dtype=torch.float32, device=self.device)

    def _raster_kwargs(self, p):
        """Common kwargs for rasterization calls."""
        scales = self.gs["scales"]
        if dpg.get_value(TAG_SCALE_CLIP_MIN_ENABLE):
            scales = scales.clamp(min=dpg.get_value(TAG_SCALE_CLIP_MIN_VALUE))
        if dpg.get_value(TAG_SCALE_CLIP_MAX_ENABLE):
            scales = scales.clamp(max=dpg.get_value(TAG_SCALE_CLIP_MAX_VALUE))
        return dict(
            means=self.gs["means"],
            quats=self.gs["quats"],
            scales=scales,
            near_plane=p["near_plane"],
            far_plane=p["far_plane"],
            radius_clip=p["radius_clip"],
            eps2d=p["eps2d"],
            rasterize_mode=p["rasterize_mode"],
            backgrounds=self._get_background(),
            width=self.render_w,
            height=self.render_h,
        )

    def _update_texture(self, image: np.ndarray):
        np.copyto(self._tex_data, image.ravel())
        dpg.set_value(TAG_RENDER_TEX, self._tex_data)

    def _update_visibility(self, meta):
        radii = meta["radii"]
        self._n_visible = int((radii.squeeze(0).amax(dim=-1) > 0).sum().item())

    def _count_contributing_from_alphas(self, opacities, render_alphas):
        """Backward through render_alphas to count Gaussians with blending weight > 0."""
        render_alphas.sum().backward()
        self._n_contributing = int((opacities.grad.abs() > 0).sum().item())

    def _compute_contributions(self, p):
        """Compute per-Gaussian contribution from ref camera via backward pass."""
        ref_name = dpg.get_value(TAG_WEIGHT_REF_CAM)
        ref_cam = self.loaded_cameras.get(ref_name)
        if ref_cam is None:
            return None
        kw = self._raster_kwargs(p)
        ref_w2c = torch.from_numpy(ref_cam["w2c"]).float().to(self.device).unsqueeze(0)
        ref_K = torch.from_numpy(ref_cam["K"]).float().to(self.device).unsqueeze(0)
        opacities = self.gs["opacities"].detach().clone().requires_grad_(True)
        _, render_alphas, _ = rasterization(
            **kw, opacities=opacities, colors=self.gs["colors"],
            viewmats=ref_w2c, Ks=ref_K, sh_degree=None, render_mode="RGB",
        )
        render_alphas.sum().backward()
        contributions = (opacities.detach() * opacities.grad.detach()).abs()
        c_min_f, c_max_f = float(contributions.min()), float(contributions.max())
        dpg.configure_item(TAG_WEIGHT_FILTER_THRESH,
                           min_value=c_min_f, max_value=c_max_f)
        return contributions

    def _apply_weight_filter(self, opacities):
        """Apply weight filter mask to opacities if enabled."""
        if self._contributions is None or not dpg.get_value(TAG_WEIGHT_FILTER_ENABLE):
            return opacities
        threshold = dpg.get_value(TAG_WEIGHT_FILTER_THRESH)
        keep_high = dpg.get_value(TAG_WEIGHT_FILTER_MODE) == "High"
        mask = self._contributions >= threshold if keep_high else self._contributions <= threshold
        return opacities * mask.float()

    def render_frame(self):
        """Rasterize Gaussians and update the DearPyGui texture."""
        if self.gs is None:
            return

        if self.gs["means"].shape[0] == 0:
            bg = self._get_background().cpu().numpy()
            image = np.full((self.render_h, self.render_w, 3), bg, dtype=np.float32)
            self._update_texture(image)
            self._n_visible = 0
            self._n_contributing = 0
            return

        cam_sel = dpg.get_value(TAG_CAMERA_SELECT)

        if cam_sel != "Orbit" and cam_sel in self.loaded_cameras:
            cam = self.loaded_cameras[cam_sel]
            w2c = cam["w2c"]
            K = cam["K"]
        else:
            c2w = self.camera.get_pose()
            w2c = np.linalg.inv(c2w).astype(np.float32)
            K = self.camera.get_intrinsics()

        w2c_t = torch.from_numpy(w2c).float().to(self.device).unsqueeze(0)
        K_t = torch.from_numpy(K).float().to(self.device).unsqueeze(0)

        p = self.render_params
        shader = p.get("shader", "RGB")

        # compute per-Gaussian contributions when needed (Weights shader or filter active)
        filter_on = dpg.get_value(TAG_WEIGHT_FILTER_ENABLE) and bool(self.loaded_cameras)
        self._contributions = None
        if shader == "Weights" or filter_on:
            self._contributions = self._compute_contributions(p)

        if shader == "Weights":
            self._render_weights(w2c_t, K_t, p)
        elif shader == "Alphas":
            self._render_alphas(w2c_t, K_t, p)
        elif shader == "Depth (expected)":
            self._render_depth(w2c_t, K_t, p, mode="RGB+ED")
        elif shader == "Depth (accumulated)":
            self._render_depth(w2c_t, K_t, p, mode="RGB+D")
        else:
            self._render_rgb(w2c_t, K_t, p)

    def _render_rgb(self, w2c_t, K_t, p):
        sh_degree = p["sh_degree"]
        if sh_degree is not None and self.gs["sh_coeffs"] is not None:
            colors = self.gs["sh_coeffs"]
        else:
            colors = self.gs["colors"]
            sh_degree = None

        opacities = self._apply_weight_filter(self.gs["opacities"]).detach().clone().requires_grad_(True)
        render_colors, render_alphas, meta = rasterization(
            **self._raster_kwargs(p),
            opacities=opacities,
            colors=colors,
            viewmats=w2c_t, Ks=K_t,
            sh_degree=sh_degree,
            render_mode="RGB",
        )
        self._update_visibility(meta)
        self._count_contributing_from_alphas(opacities, render_alphas)
        image = render_colors.detach().squeeze(0).clamp(0.0, 1.0).cpu().numpy()
        self._update_texture(image)

    def _render_alphas(self, w2c_t, K_t, p):
        """Heatmap of accumulated per-pixel blending weight (alpha)."""
        opacities = self._apply_weight_filter(self.gs["opacities"]).detach().clone().requires_grad_(True)
        _, render_alphas, meta = rasterization(
            **self._raster_kwargs(p),
            opacities=opacities,
            colors=self.gs["colors"],
            viewmats=w2c_t, Ks=K_t,
            sh_degree=None,
            render_mode="RGB",
        )
        self._update_visibility(meta)
        self._count_contributing_from_alphas(opacities, render_alphas)
        alpha = render_alphas.detach().squeeze(0).clamp(0.0, 1.0).cpu().numpy()
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        alpha_u8 = (alpha * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(alpha_u8, cv2.COLORMAP_TURBO)
        image = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self._update_texture(image)

    def _render_depth(self, w2c_t, K_t, p, mode: str = "RGB+ED"):
        """Depth heatmap. mode='RGB+ED' for expected depth, 'RGB+D' for accumulated."""
        opacities = self._apply_weight_filter(self.gs["opacities"]).detach().clone().requires_grad_(True)
        render_colors, render_alphas, meta = rasterization(
            **self._raster_kwargs(p),
            opacities=opacities,
            colors=self.gs["colors"],
            viewmats=w2c_t, Ks=K_t,
            sh_degree=None,
            render_mode=mode,
        )
        self._update_visibility(meta)
        self._count_contributing_from_alphas(opacities, render_alphas)
        # last channel is depth: (1, H, W, 4) -> (H, W)
        depth = render_colors.detach().squeeze(0)[..., -1].cpu().numpy()
        # mask out background (depth == 0)
        valid = depth > 0
        if valid.any():
            d_min = depth[valid].min()
            d_max = depth[valid].max()
            d_norm = np.where(valid, (depth - d_min) / (d_max - d_min + 1e-8), 0.0)
        else:
            d_norm = np.zeros_like(depth)
        d_u8 = (d_norm * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
        image = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # black out background
        image[~valid] = 0.0
        self._update_texture(image)

    def _render_weights(self, w2c_t, K_t, p):
        """Render per-Gaussian contribution heatmap. Contributions pre-computed in render_frame."""
        contributions = self._contributions
        if contributions is None:
            return
        kw = self._raster_kwargs(p)

        # map contributions to turbo heatmap colors (N, 3)
        c_min_f, c_max_f = float(contributions.min()), float(contributions.max())
        c_norm = (contributions - c_min_f) / (c_max_f - c_min_f + 1e-8)
        c_u8 = (c_norm * 255).byte().cpu().numpy()
        heatmap_bgr = cv2.applyColorMap(c_u8.reshape(-1, 1), cv2.COLORMAP_TURBO)
        heatmap_rgb = np.ascontiguousarray(heatmap_bgr[:, 0, ::-1]).astype(np.float32) / 255.0
        weight_colors = torch.from_numpy(heatmap_rgb).to(self.device)

        # weight filtering: opacity = 1 for kept, 0 for filtered
        opac_render = torch.ones_like(self.gs["opacities"])
        if dpg.get_value(TAG_WEIGHT_FILTER_ENABLE):
            threshold = dpg.get_value(TAG_WEIGHT_FILTER_THRESH)
            keep_high = dpg.get_value(TAG_WEIGHT_FILTER_MODE) == "High"
            mask = contributions >= threshold if keep_high else contributions <= threshold
            opac_render = mask.float()

        # render + count contributing from current view
        opac_render = opac_render.detach().clone().requires_grad_(True)
        render_colors, render_alphas, meta = rasterization(
            **kw,
            opacities=opac_render,
            colors=weight_colors,
            viewmats=w2c_t, Ks=K_t,
            sh_degree=None,
            render_mode="RGB",
        )
        self._update_visibility(meta)
        self._count_contributing_from_alphas(opac_render, render_alphas)
        image = render_colors.detach().squeeze(0).clamp(0.0, 1.0).cpu().numpy()
        self._update_texture(image)

    # ---------- main loop ----------

    def run(self):
        frame_count = 0
        fps_timer = time.time()

        while dpg.is_dearpygui_running():
            self._handle_keyboard()
            self._sync_params_from_gui()
            self.render_frame()

            dpg.render_dearpygui_frame()

            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                if self.gs is not None:
                    n_total = self.gs["means"].shape[0]
                    n_vis = getattr(self, "_n_visible", 0)
                    n_contrib = getattr(self, "_n_contributing", 0)
                    dpg.set_value(TAG_INFO_TEXT,
                                  f"{fps:.1f} FPS | {self.render_w}x{self.render_h}\n"
                                  f"{n_total:,} gaussians\n"
                                  f"{n_vis:,} visible ({100*n_vis/max(n_total,1):.1f}%)\n"
                                  f"{n_contrib:,} contributing ({100*n_contrib/max(n_vis,1):.1f}%)")
                else:
                    dpg.set_value(TAG_INFO_TEXT,
                                  f"{fps:.1f} FPS | {self.render_w}x{self.render_h}\n"
                                  f"No PLY loaded")
                dpg.set_viewport_title(f"gsplat viewer | {fps:.1f} FPS")
                frame_count = 0
                fps_timer = time.time()

        dpg.destroy_context()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time Gaussian Splatting viewer using gsplat")
    parser.add_argument("ply_path", nargs="?", type=str, default=None,
                        help="Path to a .ply file (optional, can load from GUI)")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Path to a cameras JSON file (context/target format)")
    parser.add_argument("--width", type=int, default=1920, help="Render width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Render height (default: 1080)")
    args = parser.parse_args()

    viewer = GsplatViewer(args.ply_path, cameras_path=args.cameras,
                          width=args.width, height=args.height)
    viewer.run()


if __name__ == "__main__":
    main()
