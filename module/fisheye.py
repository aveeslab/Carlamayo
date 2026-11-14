"""Kannala-Brandt fisheye cameras on CARLA 0.9.16.

CARLA's official Alpamayo example renders every PhysicalAI-AV camera with the UE5-only
``sensor.camera.rt_lens`` in its calibrated Kannala-Brandt (KB) projection,
``r = f * (θ + k1 θ³ + k2 θ⁵ + k3 θ⁷ + k4 θ⁹)``. CARLA 0.9.16 only has pinhole cameras,
so each camera renders a pinhole wide enough to hold every ray of the calibrated
1920x1080 KB image and ``FisheyeLens.remap`` resamples it into that image.
"""

import math

import cv2
import numpy as np

from . import config as cfg

# Width in pixels at which the rig's focal lengths and principal points were calibrated.
CALIBRATION_WIDTH_PX = 1920


def kb_radius(theta, cam):
    """Image radius (calibration pixels) of rays ``theta`` rad off the optical axis."""
    k1, k2, k3, k4 = cam["kb"]
    t2 = np.square(theta)
    return cam["focal_px"] * theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))


def kb_project(cam, forward, right, down, width=cfg.IMG_WIDTH):
    """Pixel ``(u, v)`` of camera-frame rays in a ``width``-wide KB image of ``cam``."""
    scale = width / CALIBRATION_WIDTH_PX
    rho = np.hypot(right, down)
    theta = np.arctan2(rho, forward)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(rho > 0, kb_radius(theta, cam) * scale / rho, 0.0)
    return cam["cx"] * scale + k * right, cam["cy"] * scale + k * down


class FisheyeLens:
    """Pinhole render size for one rig camera and its remap into the KB image.

    ``fov``, ``width`` and ``height`` are the pinhole blueprint attributes. The render
    is capped at ``max_render_width``; below that the pinhole matches the KB focal
    length at the image center.
    """

    def __init__(
        self, cam, width=cfg.IMG_WIDTH, height=cfg.IMG_HEIGHT,
        max_render_width=cfg.FISHEYE_RENDER_MAX_WIDTH,
    ):
        scale = width / CALIBRATION_WIDTH_PX
        # Invert r(θ) on a table, cut where r stops increasing (the KB fit turns over
        # beyond the calibrated field); pixels past that are left black.
        theta_table = np.linspace(0.0, math.radians(85.0), 4096)
        r_table = kb_radius(theta_table, cam) * scale
        turn = np.flatnonzero(np.diff(r_table) <= 0)
        end = turn[0] + 1 if len(turn) else len(r_table)
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        dx, dy = u - cam["cx"] * scale, v - cam["cy"] * scale
        r = np.hypot(dx, dy)
        theta = np.interp(r, r_table[:end], theta_table[:end], right=np.nan)
        # Pinhole image-plane coordinates (x right, y down at unit depth) of each ray.
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(r > 0, np.tan(theta) / r, 0.0)
        tx, ty = k * dx, k * dy
        half_w = 1.01 * np.nanmax(np.abs(tx))
        half_h = 1.01 * np.nanmax(np.abs(ty))
        self.fov = math.degrees(2.0 * math.atan(half_w))
        self.width = min(max_render_width, 2 * math.ceil(cam["focal_px"] * scale * half_w))
        focal = self.width / (2.0 * half_w)
        self.height = 2 * math.ceil(focal * half_h)
        self.map_x = np.nan_to_num(self.width / 2 + focal * tx - 0.5, nan=-1.0).astype(np.float32)
        self.map_y = np.nan_to_num(self.height / 2 + focal * ty - 0.5, nan=-1.0).astype(np.float32)

    def remap(self, image):
        """Resample a rendered pinhole image into the calibrated KB image."""
        return cv2.remap(
            image, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
