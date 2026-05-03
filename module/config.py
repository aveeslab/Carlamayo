"""Configuration for CARLA closed-loop Alpamayo pipeline."""

import os

# User Config (Edit for your local CARLA version/layout)
# Used only when CARLA_ROOT/CARLA_HOME env vars are not set.
CARLA_AGENT_ROOT = os.path.expanduser("~/carla")

# Alpamayo Configuration
NUM_CAMERAS = 4
IMG_HEIGHT = 1080
IMG_WIDTH = 1920
IMG_CHANNELS = 3
# Offscreen Epic rendering can flicker heavily with CARLA camera postprocess bloom/exposure.
CAMERA_ENABLE_POSTPROCESS_EFFECTS = False
NUM_HISTORY = 16
NUM_FRAMES = 4
NUM_TRAJ_SAMPLES = 1

# Video Configuration
SAVE_VIDEO = True
OUTPUT_VIDEO = "carla_alpamayo_closed_loop_result.mp4"
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

# Control config
CONTROL_DT = 0.1
THROTTLE_MAX = 0.35
BRAKE_MAX = 1.0
CONTROL_SMOOTH_ALPHA = 0.25

# Auto-respawn config
RESPAWN_STUCK_FRAMES = 40
RESPAWN_STUCK_SPEED_KMH = 0.5
RESPAWN_COLLISION_COOLDOWN_FRAMES = 30

# Official PID follower config
PID_LOOKAHEAD_MIN_M = 4.0
PID_LOOKAHEAD_MAX_M = 12.0
PID_LOOKAHEAD_SPEED_GAIN = 0.4
PID_TARGET_SPEED_MIN_KMH = 10.0
PID_TARGET_SPEED_MAX_KMH = 35.0
PID_TARGET_SPEED_EXTENT_GAIN = 0.5
PID_LAT_KP = 1.1
PID_LAT_KI = 0.02
PID_LAT_KD = 0.15
PID_LON_KP = 0.6
PID_LON_KI = 0.05
PID_LON_KD = 0.0
