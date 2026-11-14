"""Configuration for CARLA closed-loop Alpamayo pipeline."""

# Alpamayo Configuration
# The camera count and rig are version-specific and come from the selected
# ``--version`` adapter (module.adapters); they are no longer fixed here.
IMG_HEIGHT = 1080
IMG_WIDTH = 1920
IMG_CHANNELS = 3
# Cap on the pinhole render width behind each Kannala-Brandt camera (module.fisheye).
# The 120-degree cameras need ~4550 px for the full center focal length; 2880 keeps
# them at ~0.65x of it, still about twice the resolution the models consume.
FISHEYE_RENDER_MAX_WIDTH = 2880
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
# Throttle while the first observation window fills after a spawn or respawn (the
# official example's bootstrap), so the first plan does not start from rest.
BOOTSTRAP_THROTTLE = 0.25

# Async closed loop: brake instead of following a prediction older than this
# (simulation seconds since its inputs were captured).
TRAJECTORY_MAX_AGE_SEC = 3.0

# Auto-respawn after collision.
RESPAWN_COLLISION_COOLDOWN_FRAMES = 10

# Trajectory tracker (module.trajectory_follower), the defaults of CARLA's official
# Alpamayo example: pure-pursuit steering from the rear axle to the first plan point
# TRACKER_LOOKAHEAD_M past the nearest one, normalized by TRACKER_MAX_WHEEL_ANGLE_DEG and
# rate-limited; a PID on the plan's speed at the current trajectory time step. A
# planned speed at or below TRACKER_STOP_TARGET_SPEED_MPS is a stop: brake at least
# TRACKER_MIN_STOP_BRAKE while moving, TRACKER_HOLD_BRAKE below TRACKER_HOLD_SPEED_MPS.
TRACKER_LOOKAHEAD_M = 3.0
TRACKER_MAX_WHEEL_ANGLE_DEG = 70.0
TRACKER_STEERING_GAIN = 1.0
TRACKER_MAX_STEER_RATE_PER_S = 1.0
TRACKER_SPEED_KP = 0.35
TRACKER_SPEED_KI = 0.04
TRACKER_SPEED_KD = 0.02
TRACKER_SPEED_INTEGRAL_LIMIT = 5.0
TRACKER_STOP_TARGET_SPEED_MPS = 0.25
TRACKER_HOLD_SPEED_MPS = 0.15
TRACKER_MIN_STOP_BRAKE = 0.25
TRACKER_HOLD_BRAKE = 1.0
