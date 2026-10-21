"""Canonical CARLA camera rigs for the Alpamayo releases.

Camera keys are the canonical Alpamayo camera names and are listed in ascending
canonical camera-index order, which is the order the model consumes on the camera
axis of ``image_frames``.

Poses are NVIDIA's public Alpamayo 2 validation calibration of the seven-camera
PhysicalAI-AV rig, as used by CARLA's official Alpamayo example
(``PythonAPI/examples/av_stacks/alpamayo/alpamayo_rig.py``). They are stored in
CARLA's left-handed convention relative to the rear axle on the ground: ``x``
forward, ``y`` right, ``z`` up, ``pitch`` up, ``yaw`` clockwise seen from above,
``roll`` right side down. ``module.carla_interface.camera_transforms`` adds the ego
rear-axle offset at spawn time.

CARLA 0.9.16 renders a pinhole lens, not the source f-theta fisheye. ``fov`` is the
horizontal field of view of the pinhole whose focal length equals the calibrated
f-theta focal length, so objects near the image center appear at the real camera's
scale (a 120-degree pinhole shrinks them by ~1.7x); the outer f-theta periphery is
not rendered.
"""

import math

# Width in pixels at which the focal lengths below were calibrated.
CALIBRATION_WIDTH_PX = 1920


def _cam(x, y_left, z, yaw_left, pitch_up, focal_px, roll_right_up):
    """Convert one calibration entry (FLU, named by positive direction) to CARLA signs."""
    return {
        "x": x,
        "y": -y_left,
        "z": z,
        "pitch": pitch_up,
        "yaw": -yaw_left,
        "roll": -roll_right_up,
        "fov": math.degrees(2.0 * math.atan(CALIBRATION_WIDTH_PX / 2.0 / focal_px)),
    }


# Alpamayo 2 consumes the full seven-camera PhysicalAI source ring; each task then
# selects its fixed six-camera profile from this ring.
SEVEN_CAMERA_RIG = {
    "camera_cross_left_120fov": _cam(
        2.472564, 0.937820, 0.917814, 66.083303, -0.909268, 921.978679, 0.289044
    ),
    "camera_front_wide_120fov": _cam(
        1.696904, -0.010188, 1.435701, -0.489230, -0.443195, 925.540742, -0.155228
    ),
    "camera_cross_right_120fov": _cam(
        2.478202, -0.954542, 0.928658, -66.837341, 1.410818, 917.283211, 0.175652
    ),
    "camera_rear_left_70fov": _cam(
        2.028755, 1.039446, 0.885386, 151.680637, -3.134268, 1570.851575, -0.666026
    ),
    "camera_rear_tele_30fov": _cam(
        0.382798, 0.300216, 1.433513, 179.318246, -0.600805, 3682.876071, 0.272256
    ),
    "camera_rear_right_70fov": _cam(
        2.007933, -1.005456, 0.896404, -162.200838, -1.211118, 1560.629773, -0.535919
    ),
    "camera_front_tele_30fov": _cam(
        1.667030, 0.073855, 1.443884, -0.872023, 0.575408, 3687.248370, 0.244831
    ),
}
SEVEN_CAMERA_INDICES = (0, 1, 2, 3, 4, 5, 6)

# Alpamayo 1 (R1) and 1.5 both consume this four-camera subset: canonical indices
# [0, 1, 2, 6] = cross-left, front-wide, cross-right, front-tele.
FOUR_CAMERA_RIG = {
    name: SEVEN_CAMERA_RIG[name]
    for name in (
        "camera_cross_left_120fov",
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
    )
}
FOUR_CAMERA_INDICES = (0, 1, 2, 6)

# The calibrated forward-center poses sit inside the ego's roof/windshield, so they
# are raised just above the vehicle bounding box at spawn time.
BODY_CLEARANCE_CAMERAS = ("camera_front_wide_120fov", "camera_front_tele_30fov")

# The front-wide camera sits at position 1 in both rigs; it is the view used for
# trajectory projection overlays.
FRONT_WIDE_SLOT = 1
