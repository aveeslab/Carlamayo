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

Lens values are the same calibration's Kannala-Brandt fit of each f-theta lens
(focal length and principal point in pixels at 1920x1080). CARLA 0.9.16 has no
fisheye camera, so ``module.fisheye.FisheyeLens`` renders a wide pinhole and remaps
it into this lens.
"""


def _cam(x, y_left, z, yaw_left, pitch_up, focal_px, roll_right_up, principal_point, kb):
    """Convert one calibration entry (FLU, named by positive direction) to CARLA signs."""
    return {
        "x": x,
        "y": -y_left,
        "z": z,
        "pitch": pitch_up,
        "yaw": -yaw_left,
        "roll": -roll_right_up,
        "focal_px": focal_px,
        "cx": principal_point[0],
        "cy": principal_point[1],
        "kb": kb,
    }


# Alpamayo 2 consumes the full seven-camera PhysicalAI source ring; each task then
# selects its fixed six-camera profile from this ring.
SEVEN_CAMERA_RIG = {
    "camera_cross_left_120fov": _cam(
        2.472564, 0.937820, 0.917814, 66.083303, -0.909268, 921.978679,
        0.289044,
        (965.79980657, 546.77842273),
        (-0.020325173, 0.015561344, -0.019556958, 0.008121544),
    ),
    "camera_front_wide_120fov": _cam(
        1.696904, -0.010188, 1.435701, -0.489230, -0.443195, 925.540742,
        -0.155228,
        (957.85292258, 537.02554301),
        (-0.041555584, 0.067146887, -0.078102726, 0.031954250),
    ),
    "camera_cross_right_120fov": _cam(
        2.478202, -0.954542, 0.928658, -66.837341, 1.410818, 917.283211,
        0.175652,
        (960.73577164, 539.01147371),
        (-0.007380577, -0.015709779, 0.017383842, -0.007038902),
    ),
    "camera_rear_left_70fov": _cam(
        2.028755, 1.039446, 0.885386, 151.680637, -3.134268, 1570.851575,
        -0.666026,
        (976.85215933, 562.69828079),
        (-0.033947816, 0.401793972, -1.702209054, 2.126361486),
    ),
    "camera_rear_tele_30fov": _cam(
        0.382798, 0.300216, 1.433513, 179.318246, -0.600805, 3682.876071,
        0.272256,
        (951.05683958, 548.15984186),
        (0.231522708, -8.797345185, 159.2215684, -1036.335972),
    ),
    "camera_rear_right_70fov": _cam(
        2.007933, -1.005456, 0.896404, -162.200838, -1.211118, 1560.629773,
        -0.535919,
        (970.96068574, 549.93512723),
        (0.009383454, 0.079461877, -0.582353954, 0.775536694),
    ),
    "camera_front_tele_30fov": _cam(
        1.667030, 0.073855, 1.443884, -0.872023, 0.575408, 3687.248370,
        0.244831,
        (966.87830919, 548.33546092),
        (0.058971681, -1.359812221, 23.21701466, -149.2217783),
    ),
}
SEVEN_CAMERA_INDICES = (0, 1, 2, 3, 4, 5, 6)

# Alpamayo 2 Edge consumes every camera of this ring except the rear tele.
SIX_CAMERA_RIG = {
    name: cam for name, cam in SEVEN_CAMERA_RIG.items() if name != "camera_rear_tele_30fov"
}
SIX_CAMERA_INDICES = (0, 1, 2, 3, 5, 6)

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
