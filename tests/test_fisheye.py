import math

import cv2
import numpy as np
import pytest

from module import config as cfg
from module.adapters._rigs import SEVEN_CAMERA_RIG
from module.fisheye import FisheyeLens, kb_project


def test_every_rig_lens_is_rendered_from_a_pinhole_that_covers_the_image():
    for name, cam in SEVEN_CAMERA_RIG.items():
        lens = FisheyeLens(cam)
        assert lens.width <= cfg.FISHEYE_RENDER_MAX_WIDTH, name
        assert lens.map_x.shape == (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), name
        assert 0 <= lens.map_x.min() and lens.map_x.max() <= lens.width - 1, name
        assert 0 <= lens.map_y.min() and lens.map_y.max() <= lens.height - 1, name
        if "120fov" in name:
            assert 133.0 < lens.fov < 137.0, name


def test_remap_moves_a_rendered_ray_to_its_kannala_brandt_pixel():
    cam = SEVEN_CAMERA_RIG["camera_front_wide_120fov"]
    lens = FisheyeLens(cam)
    focal = lens.width / (2 * math.tan(math.radians(lens.fov / 2)))
    assert kb_project(cam, 1.0, 0.0, 0.0) == pytest.approx((cam["cx"], cam["cy"]))
    for right, down in [(0.3, 0.1), (-0.9, 0.3)]:
        rendered = np.zeros((lens.height, lens.width), np.uint8)
        center = lens.width / 2 + focal * right - 0.5, lens.height / 2 + focal * down - 0.5
        cv2.circle(rendered, (round(center[0]), round(center[1])), 3, 255, -1)
        ys, xs = np.nonzero(lens.remap(rendered) > 128)
        u, v = kb_project(cam, 1.0, right, down)
        assert (xs.mean(), ys.mean()) == pytest.approx((u, v), abs=1.5)


def test_pixels_beyond_where_the_lens_fit_turns_over_are_black():
    cam = {**SEVEN_CAMERA_RIG["camera_front_wide_120fov"], "kb": (-0.6, 0.0, 0.0, 0.0)}
    lens = FisheyeLens(cam)
    out = lens.remap(np.full((lens.height, lens.width), 255, np.uint8))
    assert out[0, 0] == 0 and out[cfg.IMG_HEIGHT // 2, cfg.IMG_WIDTH // 2] == 255
