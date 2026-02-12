"""Self-contained orbit camera for the gsplat viewer.

Provides a Blender-like turntable orbit camera with:
  - Orbit (left-click drag)
  - Pan (right-click / Shift+left drag)
  - Zoom (scroll wheel)
  - WASD movement
  - Preset views (front, back, top, etc.)
"""

import numpy as np
import math

EPS = 1e-6
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def _normalize(v):
    n = float(np.linalg.norm(v))
    return v / max(n, EPS)


def _look_at(eye, center, up):
    """
    Camera pose (camera -> world) in a right-handed system.

    Columns are the camera's basis vectors in world space:
    [ right | up | forward | eye ]
    where forward points from the camera toward `center`.
    """
    eye = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = _normalize(center - eye)
    right = np.cross(forward, _normalize(up))
    if np.linalg.norm(right) < EPS:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(ref, forward)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, ref)
    right = _normalize(right)
    new_up = _normalize(np.cross(right, forward))

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = -new_up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def _get_intrinsics(height, width, fovy):
    f = height / (2.0 * math.tan(math.radians(fovy) * 0.5))
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = width * 0.5
    K[1, 2] = height * 0.5
    return K


def _get_pose(azimuth_deg, elevation_deg, radius, center, up):
    """Turntable spherical coordinates around `center` with Z-up or Y-up."""
    center = np.asarray(center, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    a = math.radians(float(azimuth_deg))
    e = math.radians(float(elevation_deg))
    r = float(radius)

    upz = np.allclose(up, (0.0, 0.0, 1.0), atol=1e-6)
    upy = np.allclose(up, (0.0, 1.0, 0.0), atol=1e-6)

    if upz:
        offset = np.array(
            [r * math.cos(e) * math.cos(a), r * math.cos(e) * math.sin(a), r * math.sin(e)],
            dtype=np.float32,
        )
    elif upy:
        offset = np.array(
            [r * math.cos(e) * math.cos(a), r * math.sin(e), r * math.cos(e) * math.sin(a)],
            dtype=np.float32,
        )
    else:
        w = _normalize(up)
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(ref, w)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u = _normalize(np.cross(w, ref))
        v = _normalize(np.cross(w, u))
        offset = r * (math.cos(e) * math.cos(a) * u + math.cos(e) * math.sin(a) * v + math.sin(e) * w).astype(
            np.float32
        )

    eye = center + offset
    return _look_at(eye, center, up)


class Camera:
    """Minimal camera class managing intrinsics and c2w pose."""

    def __init__(self, intrinsics, pose, width, height, near=0.1, far=10000.0):
        assert intrinsics.shape == (3, 3)
        assert pose.shape == (4, 4)
        self.set_intrinsics(intrinsics)
        self.pose = pose.astype(np.float32)
        self.width = width
        self.height = height
        self.near = near
        self.far = far

    def set_intrinsics(self, intrinsics):
        self.intrinsics = intrinsics.astype(np.float32)
        self.intrinsics_inv = np.linalg.inv(intrinsics)

    def get_intrinsics(self):
        return self.intrinsics

    def get_pose(self):
        return self.pose

    def get_pose_inv(self):
        return np.linalg.inv(self.get_pose())


class OrbitCamera(Camera):
    """
    Blender-like turntable orbit camera.

    Controls:
        - Orbit:       LMB drag -> orbit(dx_px, dy_px)
        - Pan:         RMB / Shift+LMB drag -> pan(dx_px, dy_px)
        - Zoom:        Scroll wheel -> zoom(scroll_steps)
        - WASD:        move_forward/backward/left/right(fast)
        - Quick views: set_view('front'|'back'|'right'|'left'|'top'|'bottom')
    """

    def __init__(
        self,
        width: int,
        height: int,
        radius: float = 3.0,
        fovy: float = 50.0,
        near: float = 0.01,
        far: float = 1000.0,
        center: np.ndarray = np.array([0, 0, 0], dtype=np.float32),
        up: str = "z",
        azimuth_deg: float = 45.0,
        elevation_deg: float = 30.0,
    ):
        if up == "z":
            upv = np.array([0, 0, 1], dtype=np.float32)
        elif up == "y":
            upv = np.array([0, 1, 0], dtype=np.float32)
        else:
            raise ValueError(f"Invalid `up` value: {up}, must be 'z' or 'y'.")

        intrinsics = _get_intrinsics(height, width, fovy)
        pose = _get_pose(azimuth_deg, elevation_deg, radius, center, upv)

        super().__init__(
            intrinsics=intrinsics, pose=pose,
            width=width, height=height, near=near, far=far,
        )

        self.radius = float(radius)
        self.fovy = float(fovy)
        self.center = np.array(center, dtype=np.float32)
        self.up_world = upv
        self.azimuth_deg = float(azimuth_deg)
        self.elevation_deg = float(elevation_deg)

        # tunables
        self.min_radius = 1e-3
        self.max_radius = 1e6
        self.min_pitch = -89.5
        self.max_pitch = 89.5
        self.rotate_speed = 0.5
        self.pan_speed = 0.5
        self.move_speed = 0.5
        self.move_speed_fast = self.move_speed * 5
        self.dolly_per_scroll = 1.2
        self.dolly_drag_sensitivity = 1.8

        self._update()

    # ---------- internals ----------

    @property
    def aspect(self):
        return float(self.width) / max(1.0, float(self.height))

    def _update(self):
        self.azimuth_deg = (self.azimuth_deg + 360.0) % 360.0
        self.elevation_deg = float(np.clip(self.elevation_deg, self.min_pitch, self.max_pitch))
        self.radius = float(np.clip(self.radius, self.min_radius, self.max_radius))
        intrinsics = _get_intrinsics(self.height, self.width, self.fovy)
        self.set_intrinsics(intrinsics)
        self.pose = _get_pose(self.azimuth_deg, self.elevation_deg, self.radius, self.center, self.up_world)

    def _spherical_eye(self):
        a = self.azimuth_deg * DEG2RAD
        e = self.elevation_deg * DEG2RAD
        if np.allclose(self.up_world, (0, 0, 1), atol=1e-8):
            x = self.radius * np.cos(e) * np.cos(a)
            y = self.radius * np.cos(e) * np.sin(a)
            z = self.radius * np.sin(e)
        else:
            x = self.radius * np.cos(e) * np.cos(a)
            y = self.radius * np.sin(e)
            z = self.radius * np.cos(e) * np.sin(a)
        return self.center + np.array([x, y, z], dtype=np.float32)

    def _camera_axes(self):
        eye = self._spherical_eye()
        fwd = self.center - eye
        fwd /= np.linalg.norm(fwd) + EPS
        right = np.cross(fwd, self.up_world)
        n = np.linalg.norm(right)
        if n < EPS:
            arbitrary = np.array([1, 0, 0], dtype=np.float32)
            if abs(np.dot(arbitrary, self.up_world)) > 0.9:
                arbitrary = np.array([0, 1, 0], dtype=np.float32)
            right = np.cross(fwd, arbitrary)
            right /= np.linalg.norm(right) + EPS
        else:
            right /= n
        up_cam = np.cross(right, fwd)
        up_cam /= np.linalg.norm(up_cam) + EPS
        return right.astype(np.float32), up_cam.astype(np.float32), fwd.astype(np.float32)

    def _view_extents_at_target(self, radius=None):
        if radius is None:
            radius = self.radius
        fovy_rad = self.fovy * DEG2RAD
        h = 2.0 * radius * np.tan(0.5 * fovy_rad)
        w = h * self.aspect
        return w, h

    # ---------- public setters ----------

    def set_center(self, center):
        self.center = np.array(center, dtype=np.float32)
        self._update()

    def set_fov(self, fovy):
        self.fovy = float(fovy)
        self._update()

    def set_elevation_deg(self, elevation_deg):
        self.elevation_deg = float(np.clip(elevation_deg, self.min_pitch + EPS, self.max_pitch - EPS))
        self._update()

    def set_azimuth_deg(self, azimuth_deg):
        self.azimuth_deg = float(azimuth_deg)
        self._update()

    def set_radius(self, radius):
        self.radius = float(np.clip(radius, self.min_radius, self.max_radius))
        self._update()

    # ---------- controls ----------

    def orbit(self, dx_px, dy_px):
        yaw_delta = (dx_px / max(1.0, self.width)) * (360.0 * self.rotate_speed)
        pitch_delta = (dy_px / max(1.0, self.height)) * (180.0 * self.rotate_speed)
        self.azimuth_deg += yaw_delta
        self.elevation_deg -= pitch_delta
        self._update()

    def pan(self, dx_px, dy_px):
        right, up_cam, _ = self._camera_axes()
        view_w, view_h = self._view_extents_at_target(self.radius)
        tx = (dx_px / max(1.0, self.width)) * view_w * self.pan_speed
        ty = (dy_px / max(1.0, self.height)) * view_h * self.pan_speed
        self.center -= right * tx
        self.center += up_cam * ty
        self._update()

    def zoom(self, scroll_steps, mouse_xy=None):
        if scroll_steps == 0:
            return
        old_radius = self.radius
        factor = self.dolly_per_scroll ** (-float(scroll_steps))
        self.radius = float(np.clip(old_radius * factor, self.min_radius, self.max_radius))
        if mouse_xy is not None:
            self._zoom_to_mouse_compensate(mouse_xy, old_radius, self.radius)
        self._update()

    def _zoom_to_mouse_compensate(self, mouse_xy, old_r, new_r):
        mx, my = float(mouse_xy[0]), float(mouse_xy[1])
        x_ndc = (mx / max(1.0, self.width)) * 2.0 - 1.0
        y_ndc = 1.0 - (my / max(1.0, self.height)) * 2.0
        right, up_cam, _ = self._camera_axes()
        old_w, old_h = self._view_extents_at_target(old_r)
        new_w, new_h = self._view_extents_at_target(new_r)
        old_off = right * (x_ndc * old_w * 0.5) + up_cam * (y_ndc * old_h * 0.5)
        new_off = right * (x_ndc * new_w * 0.5) + up_cam * (y_ndc * new_h * 0.5)
        self.center += old_off - new_off

    def focus_bounds(self, bmin, bmax, padding=1.05):
        bmin = np.array(bmin, dtype=np.float32)
        bmax = np.array(bmax, dtype=np.float32)
        c = 0.5 * (bmin + bmax)
        ext = 0.5 * (bmax - bmin)
        radius_sphere = float(np.linalg.norm(ext))
        fovy_rad = self.fovy * DEG2RAD
        fovx_rad = 2.0 * np.arctan(np.tan(fovy_rad * 0.5) * self.aspect)
        req_dist_y = radius_sphere / (np.sin(fovy_rad * 0.5) + EPS)
        req_dist_x = radius_sphere / (np.sin(fovx_rad * 0.5) + EPS)
        req_dist = max(req_dist_x, req_dist_y) * padding
        self.center = c
        self.radius = max(req_dist, self.min_radius)
        self._update()

    def set_view(self, name: str):
        n = name.lower()
        views = {
            "front": (0.0, 0.0), "back": (180.0, 0.0),
            "right": (90.0, 0.0), "left": (-90.0, 0.0),
            "top": (0.0, 90.0), "bottom": (0.0, -90.0),
        }
        if n in views:
            self.azimuth_deg, self.elevation_deg = views[n]
            self._update()

    # ---------- WASD movement ----------

    def move_forward(self, fast=False):
        speed = self.move_speed_fast if fast else self.move_speed
        d = self.center - self._spherical_eye()
        d /= np.linalg.norm(d)
        self.center += d * speed
        self._update()

    def move_backward(self, fast=False):
        speed = self.move_speed_fast if fast else self.move_speed
        d = self.center - self._spherical_eye()
        d /= np.linalg.norm(d)
        self.center -= d * speed
        self._update()

    def move_left(self, fast=False):
        speed = self.move_speed_fast if fast else self.move_speed
        right, _, _ = self._camera_axes()
        self.center -= right * speed
        self._update()

    def move_right(self, fast=False):
        speed = self.move_speed_fast if fast else self.move_speed
        right, _, _ = self._camera_axes()
        self.center += right * speed
        self._update()

    @property
    def position(self):
        return self._spherical_eye()
