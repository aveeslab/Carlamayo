"""Configuration for CARLA closed-loop Alpamayo pipeline."""

import os

# User Config (Edit for your local CARLA version/layout)
# Used only when CARLA_ROOT/CARLA_HOME env vars are not set.
CARLA_AGENT_ROOT = os.path.expanduser("~/carla")

# Alpamayo Configuration
# The camera count and rig are version-specific and come from the selected
# ``--version`` adapter (module.adapters); they are no longer fixed here.
IMG_HEIGHT = 1080
IMG_WIDTH = 1920
IMG_CHANNELS = 3
# Offscreen Epic rendering can flicker heavily with CARLA camera postprocess bloom/exposure.
CAMERA_ENABLE_POSTPROCESS_EFFECTS = True
NUM_HISTORY = 16
NUM_FRAMES = 4
NUM_TRAJ_SAMPLES = 1

# Video Configuration
SAVE_VIDEO = True
OUTPUT_VIDEO = "carla_alpamayo_closed_loop_result.mp4"
OPEN_LOOP_OUTPUT_VIDEO = "carla_alpamayo_open_loop_result.mp4"
LIVE_OPEN_LOOP_OUTPUT_VIDEO = "carla_alpamayo_live_open_loop_result.mp4"
VIDEO_FPS = 10
PYGAME_WINDOW_WIDTH = 1280
PYGAME_WINDOW_HEIGHT = 900

# CARLA Configuration
CARLA_MAP = "Town03"  # Urban-style map
NPC_VEHICLE_COUNT = 50
NPC_WALKER_COUNT = 50
NPC_EXCLUDED_VEHICLE_KEYWORDS = (
    "ambulance",
    "carlacola",
    "cybertruck",
    "firetruck",
    "fusorosa",
    "sprinter",
)

# Ego vehicle (vehicle.lincoln.mkz_2020). Alpamayo's ego frame and the calibrated
# camera rig are anchored at the rear axle; this is its x offset from the CARLA
# actor origin, measured on CARLA 0.9.16.
EGO_REAR_AXLE_X = -1.3886
# Forward-center cameras are raised this far above the ego bounding box.
CAMERA_BODY_CLEARANCE_M = 0.08

# Control config
CONTROL_DT = 0.1
THROTTLE_MAX = 0.6
BRAKE_MAX = 1.0
CONTROL_SMOOTH_ALPHA = 0.25

# Async closed loop: brake instead of following a prediction older than this
# (simulation seconds since its inputs were captured).
TRAJECTORY_MAX_AGE_SEC = 3.0

# Auto-respawn after collision.
RESPAWN_COLLISION_COOLDOWN_FRAMES = 10

# Official PID follower config
PID_LOOKAHEAD_MIN_M = 4.0
PID_LOOKAHEAD_MAX_M = 12.0
PID_LOOKAHEAD_SPEED_GAIN = 0.4
# Target speed = Alpamayo's planned speed, averaged over this much of the plan
# from now (covers actuator lag), capped at PID_TARGET_SPEED_MAX_KMH. A planned
# speed at or below PID_STOP_SPEED_MPS is treated as a stop and holds the brake.
PID_SPEED_PREVIEW_SEC = 0.5
PID_TARGET_SPEED_MAX_KMH = 35.0
PID_STOP_SPEED_MPS = 0.25
# Start-from-rest: below PID_REST_SPEED_MPS, launch at PID_LAUNCH_SPEED_MPS when the
# remaining plan still covers at least PID_LAUNCH_MIN_PLAN_M.
PID_REST_SPEED_MPS = 0.5
PID_LAUNCH_SPEED_MPS = 2.0
PID_LAUNCH_MIN_PLAN_M = 3.0
PID_LAT_KP = 1.1
PID_LAT_KI = 0.02
PID_LAT_KD = 0.15
PID_LON_KP = 0.9
PID_LON_KI = 0.05
PID_LON_KD = 0.0
