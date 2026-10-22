"""Configuration for CARLA closed-loop Alpamayo pipeline."""

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
# Seed set before every trajectory inference. Alpamayo samples its CoT (temperature
# 0.6) and diffusion noise; reusing one seed keeps consecutive plans comparable
# instead of jumping by several metres between 0.1 s ticks.
INFERENCE_SEED = 42

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
# actor origin, and the wheelbase, both measured on CARLA 0.9.16.
EGO_REAR_AXLE_X = -1.3886
EGO_WHEELBASE_M = 2.86
# Forward-center cameras are raised this far above the ego bounding box.
CAMERA_BODY_CLEARANCE_M = 0.08

# Control config
CONTROL_DT = 0.1
THROTTLE_MAX = 0.75
BRAKE_MAX = 1.0

# Async closed loop: brake instead of following a prediction older than this
# (simulation seconds since its inputs were captured).
TRAJECTORY_MAX_AGE_SEC = 3.0

# Auto-respawn after collision.
RESPAWN_COLLISION_COOLDOWN_FRAMES = 10

# Trajectory follower (module.trajectory_follower): pure-pursuit steering from the
# rear axle to a speed-dependent lookahead point, steer change rate-limited.
PID_LOOKAHEAD_MIN_M = 4.0
PID_LOOKAHEAD_MAX_M = 12.0
PID_LOOKAHEAD_SPEED_GAIN = 0.4
# Target speed = Alpamayo's planned speed, averaged over PID_SPEED_PREVIEW_SEC of the
# plan starting PID_SPEED_LEAD_SEC ahead (covers actuator lag), capped at
# PID_TARGET_SPEED_MAX_KMH. A planned
# speed at or below PID_STOP_SPEED_MPS is treated as a stop and holds the brake.
PID_SPEED_PREVIEW_SEC = 0.5
PID_SPEED_LEAD_SEC = 1.0
PID_TARGET_SPEED_MAX_KMH = 35.0
PID_STOP_SPEED_MPS = 0.25
# Start-from-rest: below PID_REST_SPEED_MPS, launch at PID_LAUNCH_SPEED_MPS when the
# remaining plan still covers at least PID_LAUNCH_MIN_PLAN_M, and keep launching while
# it covers at least PID_LAUNCH_HOLD_PLAN_M.
PID_REST_SPEED_MPS = 0.5
PID_LAUNCH_SPEED_MPS = 2.0
PID_LAUNCH_MIN_PLAN_M = 3.0
PID_LAUNCH_HOLD_PLAN_M = 1.0
FOLLOW_STEER_RATE_PER_SEC = 1.0
# Speed: throttle feedforward measured for the MKZ on a flat road (throttle that holds
# a steady speed, linear in m/s), plus PI on the m/s error with the integral clamped to
# FOLLOW_KI_EFFORT_LIMIT of pedal and derivative on measurement. The target is the
# median of the last FOLLOW_TARGET_MEDIAN_TICKS planned speeds, which drops one-tick
# dips of the replanned profile without delaying sustained changes; planned stops
# bypass it.
FOLLOW_FF_THROTTLE_AT_ZERO = 0.36
FOLLOW_FF_THROTTLE_PER_MPS = 0.0167
FOLLOW_KP = 0.35
FOLLOW_KI = 0.04
FOLLOW_KD = 0.02
FOLLOW_KI_EFFORT_LIMIT = 0.3
FOLLOW_TARGET_MEDIAN_TICKS = 3
# Braking. Measured on the MKZ: any brake value from 0.03 to 1.0 gives ~6.9 m/s^2, so
# the brake is on/off, while pedals released shed roughly 0.4 * speed m/s^2. When the
# target is below FOLLOW_BRAKE_SPEED_RATIO of the current speed, coasting cannot get
# there in time and the brake is applied at FOLLOW_BRAKE_ON until the target rises
# above FOLLOW_BRAKE_RELEASE_RATIO of the speed (hysteresis). A planned stop brakes
# while moving and holds full brake once stopped.
FOLLOW_BRAKE_SPEED_RATIO = 0.6
FOLLOW_BRAKE_RELEASE_RATIO = 0.8
FOLLOW_BRAKE_ON = 0.3
# Lateral acceleration limit used to cap speed by the plan's curvature ahead (m/s^2).
FOLLOW_LAT_ACCEL_MAX = 3.0
