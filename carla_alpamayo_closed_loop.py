"""CARLA closed-loop control with local Alpamayo inference (HQ recording)."""

import time
import math
import copy
import queue
import textwrap
import argparse
import os
import sys
import traceback
import numpy as np
import cv2
import carla
import random
import torch
from einops import rearrange
from transformers import BitsAndBytesConfig


def _resolve_vehicle_pid_controller():
    """Import VehiclePIDController, auto-adding env/relative CARLA agent paths."""
    try:
        from agents.navigation.controller import VehiclePIDController as _VehiclePIDController
        return _VehiclePIDController
    except ImportError:
        pass

    candidate_roots = []
    for env_key in ("CARLA_ROOT", "CARLA_HOME"):
        v = os.environ.get(env_key)
        if v:
            candidate_roots.append(v)

    # Repository-relative/common local paths (no user-specific absolute paths).
    candidate_roots.extend([
        "carla",
        os.path.join("carla", "CARLA_0.9.16"),
        os.path.join("..", "carla"),
        os.path.join("..", "CARLA_0.9.16"),
    ])

    for root in candidate_roots:
        agents_parent = os.path.abspath(os.path.join(root, "PythonAPI", "carla"))
        if os.path.isdir(agents_parent) and agents_parent not in sys.path:
            sys.path.append(agents_parent)

    try:
        from agents.navigation.controller import VehiclePIDController as _VehiclePIDController
        return _VehiclePIDController
    except ImportError as e:
        raise ImportError(
            "VehiclePIDController not found. Set CARLA_ROOT or add "
            "'<CARLA_ROOT>/PythonAPI/carla' to PYTHONPATH."
        ) from e

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper


# ============================================================================
# Configuration
# ============================================================================
NUM_CAMERAS = 4
IMG_HEIGHT = 1080
IMG_WIDTH = 1920
IMG_CHANNELS = 3
NUM_HISTORY = 16
NUM_FRAMES = 4
NUM_TRAJ_SAMPLES = 4
SAVE_VIDEO = True
OUTPUT_VIDEO = "carla_alpamayo_closed_loop_result_hq.mp4"
VIDEO_FPS = 10
CARLA_MAP = "Town03"  # Urban-style map
NPC_VEHICLE_COUNT = 50
NPC_WALKER_COUNT = 50

# Control config
CONTROL_DT = 0.1
THROTTLE_MAX = 0.35
BRAKE_MAX = 1.0
CONTROL_SMOOTH_ALPHA = 0.25

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CARLA closed-loop control with Alpamayo."
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Use 4-bit quantized model. Default is full-precision.",
    )
    return parser.parse_args()


# ============================================================================
# Official PID Waypoint Follower
# ============================================================================
def alpamayo_to_carla_local(wp_ego):
    """Convert Alpamayo local frame (y=left) to CARLA local frame (y=right)."""
    wp_local = np.asarray(wp_ego, dtype=np.float64).copy()
    wp_local[:, 1] *= -1.0
    return wp_local


def local_to_world(vehicle_tf, wp_local):
    """Convert local waypoints to world using CARLA transform."""
    wp_world = []
    for p in wp_local:
        loc_w = vehicle_tf.transform(carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        wp_world.append([loc_w.x, loc_w.y, loc_w.z])
    return np.asarray(wp_world, dtype=np.float64)


class OfficialPIDFollower:
    """CARLA official PID waypoint follower."""

    def __init__(self, world, vehicle):
        VehiclePIDController = _resolve_vehicle_pid_controller()
        self.world = world
        self.map = world.get_map()
        self.vehicle = vehicle
        args_lateral = {"K_P": PID_LAT_KP, "K_I": PID_LAT_KI, "K_D": PID_LAT_KD, "dt": CONTROL_DT}
        args_longitudinal = {"K_P": PID_LON_KP, "K_I": PID_LON_KI, "K_D": PID_LON_KD, "dt": CONTROL_DT}
        self.pid = VehiclePIDController(
            vehicle,
            args_lateral=args_lateral,
            args_longitudinal=args_longitudinal,
            max_throttle=THROTTLE_MAX,
            max_brake=BRAKE_MAX,
            max_steering=0.8,
        )

    def _pick_target(self, wp_world, speed_mps):
        lookahead_m = float(np.clip(
            PID_LOOKAHEAD_MIN_M + PID_LOOKAHEAD_SPEED_GAIN * speed_mps,
            PID_LOOKAHEAD_MIN_M,
            PID_LOOKAHEAD_MAX_M,
        ))
        if len(wp_world) == 0:
            return None, 0, lookahead_m
        if len(wp_world) == 1:
            target_idx = 0
        else:
            seg = np.linalg.norm(np.diff(wp_world[:, :2], axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            target_idx = int(min(np.searchsorted(cum, lookahead_m), len(wp_world) - 1))
        loc = carla.Location(
            x=float(wp_world[target_idx, 0]),
            y=float(wp_world[target_idx, 1]),
            z=float(wp_world[target_idx, 2]),
        )
        target_wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        return target_wp, target_idx, lookahead_m

    def compute_control(self, vehicle_tf, wp_ego, speed_mps):
        wp_local = alpamayo_to_carla_local(wp_ego)
        wp_world = local_to_world(vehicle_tf, wp_local)
        traj_extent = float(np.max(np.linalg.norm(wp_local[:, :2], axis=1)))
        target_speed_kmh = float(np.clip(
            PID_TARGET_SPEED_MIN_KMH + PID_TARGET_SPEED_EXTENT_GAIN * traj_extent,
            PID_TARGET_SPEED_MIN_KMH,
            PID_TARGET_SPEED_MAX_KMH,
        ))
        target_wp, target_idx, lookahead_m = self._pick_target(wp_world, speed_mps)
        if target_wp is None:
            return 0.0, 0.0, 0.0, {
                "mode": "official_pid_no_target",
                "traj_extent": traj_extent,
            }
        control = self.pid.run_step(target_speed_kmh, target_wp)
        debug = {
            "mode": "official_pid",
            "target_speed_kmh": target_speed_kmh,
            "lookahead_m": lookahead_m,
            "target_idx": int(target_idx),
            "target_wp_xy": [float(target_wp.transform.location.x), float(target_wp.transform.location.y)],
            "traj_extent": traj_extent,
        }
        return float(control.steer), float(control.throttle), float(control.brake), debug


# ============================================================================
# Visualization Functions
# ============================================================================
def _project_one_trajectory(result, points_3d, img_width, img_height, focal_length_px, camera_height, line_color, point_color, line_thickness):
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z_cam = z + camera_height
    valid = x > 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        u = img_width / 2 - (y / x) * focal_length_px
        v_temp = img_height / 2 - (z_cam / x) * focal_length_px
        v = img_height - v_temp

    u = np.clip(u, 0, img_width - 1).astype(np.int32)
    v = np.clip(v, 0, img_height - 1).astype(np.int32)
    points_2d = np.column_stack([u[valid], v[valid]])
    if len(points_2d) <= 1:
        return

    for i in range(len(points_2d) - 1):
        cv2.line(
            result,
            tuple(points_2d[i]),
            tuple(points_2d[i + 1]),
            line_color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
    for pt in points_2d:
        cv2.circle(result, tuple(pt), max(4, line_thickness), point_color, -1, cv2.LINE_AA)


def project_trajectory_to_image(cam_img, pred_xyz, selected_idx=0, camera_height=2.4, fov=120):
    """Project one or multiple trajectories onto image.

    - Selected trajectory: red (existing style)
    - Non-selected trajectories: white
    """
    img_height, img_width = cam_img.shape[:2]
    focal_length_px = img_width / (2 * np.tan(np.radians(fov / 2)))

    result = cam_img.copy()
    if isinstance(pred_xyz, torch.Tensor):
        traj_samples = extract_trajectory_samples(pred_xyz)
    else:
        arr = np.asarray(pred_xyz)
        if arr.ndim == 2:
            traj_samples = arr[None, :, :3]
        elif arr.ndim == 3:
            traj_samples = arr[:, :, :3]
        else:
            return result

    num_samples = traj_samples.shape[0]
    selected_idx = int(np.clip(selected_idx, 0, max(0, num_samples - 1)))

    # Draw non-selected first (white), then selected on top (red).
    for i in range(num_samples):
        if i == selected_idx:
            continue
        _project_one_trajectory(
            result=result,
            points_3d=traj_samples[i],
            img_width=img_width,
            img_height=img_height,
            focal_length_px=focal_length_px,
            camera_height=camera_height,
            line_color=(255, 255, 255),
            point_color=(255, 255, 255),
            line_thickness=4,
        )

    _project_one_trajectory(
        result=result,
        points_3d=traj_samples[selected_idx],
        img_width=img_width,
        img_height=img_height,
        focal_length_px=focal_length_px,
        camera_height=camera_height,
        line_color=(255, 0, 0),
        point_color=(255, 100, 100),
        line_thickness=8,
    )

    return result


def create_visualization_frame(cam_img, pred_xyz, selected_idx, frame_count, inference_time,
                                cot_text, speed_kmh, steering):
    """Create a single visualization frame with all overlays."""
    # Project trajectory
    vis_img = project_trajectory_to_image(cam_img, pred_xyz, selected_idx=selected_idx)

    # Convert to BGR for OpenCV text rendering
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # Add semi-transparent overlay boxes
    h, w = vis_img.shape[:2]

    # Bottom CoT box
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (10, h - 150), (w - 10, h - 10), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0)

    # Top text: Frame info
    info_text = f"Frame: {frame_count} | Inference: {inference_time:.2f}s | Speed: {speed_kmh:.1f} km/h | Steer: {steering:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (tw, th), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
    pad_x = 14
    pad_y = 14
    box_x1, box_y1 = 10, 10
    box_x2 = min(w - 10, box_x1 + tw + pad_x * 2)
    box_y2 = box_y1 + th + pad_y * 2
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    vis_img = cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0)

    cv2.putText(vis_img, info_text, (20, 50),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Bottom text: Chain-of-Causation
    cot_display = cot_text[:200] + "..." if len(cot_text) > 200 else cot_text
    # Wrap text
    max_chars = 120
    lines = textwrap.wrap(f"CoT: {cot_display}", width=max_chars)
    y_offset = h - 120
    for line in lines[:3]:  # Max 3 lines
        cv2.putText(vis_img, line, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 30

    # Convert back to RGB
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    return vis_img


class VideoRecorder:
    """Records frames and saves to video file."""

    def __init__(self, output_path, fps=10):
        self.output_path = output_path
        self.fps = fps
        self.frames = []

    def add_frame(self, frame):
        """Add a frame (RGB numpy array)."""
        self.frames.append(frame)

    def _create_writer(self, width, height):
        for codec in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            if writer.isOpened():
                return writer, codec
            writer.release()
        return None, None

    def save(self):
        """Save all frames to video."""
        if not self.frames:
            print("No frames to save.")
            return

        print(f"\nSaving video with {len(self.frames)} frames...")
        h, w = self.frames[0].shape[:2]
        writer, selected_codec = self._create_writer(w, h)
        if writer is None:
            print("Failed to initialize video writer.")
            return
        if hasattr(cv2, "VIDEOWRITER_PROP_QUALITY"):
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

        for frame in self.frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        print(f"Video saved: {self.output_path}")
        print(f"  Codec: {selected_codec}, Resolution: {w}x{h}, FPS: {self.fps}, Frames: {len(self.frames)}")


# ============================================================================
# CARLA Interface
# ============================================================================
class CARLAInterface:
    """Interface for CARLA simulation."""

    def __init__(self):
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.sensors = {}
        self.sensor_queues = {}
        self.history_buffer = []
        self.npc_vehicle_ids = []
        self.npc_walker_ids = []
        self.npc_walker_controller_ids = []
        self.tm_port = 8000

        self.camera_configs = {
            "cam_front_left": {"x": 1.0, "y": -0.5, "z": 2.4, "pitch": 0.0, "yaw": -60.0, "fov": 120},
            "cam_front_wide": {"x": 1.5, "y": 0.0, "z": 2.4, "pitch": 0.0, "yaw": 0.0, "fov": 95},
            "cam_front_right": {"x": 1.0, "y": 0.5, "z": 2.4, "pitch": 0.0, "yaw": 60.0, "fov": 120},
            "cam_front_tele": {"x": 1.5, "y": 0.0, "z": 2.4, "pitch": 0.0, "yaw": 0.0, "fov": 30},
        }
        self.camera_order = ["cam_front_left", "cam_front_wide", "cam_front_right", "cam_front_tele"]
        self.spawn_meta = {}

    def _count_adjacent_driving_lanes(self, waypoint, max_hops=8):
        count = 1
        visited = {waypoint.id}
        left = waypoint.get_left_lane()
        hops = 0
        while (
            left is not None
            and left.lane_type == carla.LaneType.Driving
            and left.id not in visited
            and hops < max_hops
        ):
            count += 1
            visited.add(left.id)
            left = left.get_left_lane()
            hops += 1

        hops = 0
        right = waypoint.get_right_lane()
        while (
            right is not None
            and right.lane_type == carla.LaneType.Driving
            and right.id not in visited
            and hops < max_hops
        ):
            count += 1
            visited.add(right.id)
            right = right.get_right_lane()
            hops += 1
        return count

    def _straightness_score(self, waypoint, step_dist=10.0, steps=6):
        wp = waypoint
        yaws = [wp.transform.rotation.yaw]
        for _ in range(steps):
            nxt = wp.next(step_dist)
            if not nxt:
                break
            wp = nxt[0]
            yaws.append(wp.transform.rotation.yaw)
            if wp.is_junction:
                break
        if len(yaws) < 3:
            return -999.0, len(yaws)
        y0 = yaws[0]
        max_dev = max(abs(((y - y0 + 180.0) % 360.0) - 180.0) for y in yaws)
        return -max_dev, len(yaws)

    def _select_highway_spawn_point(self, spawn_points, max_candidates=120, max_search_sec=3.0):
        carla_map = self.world.get_map()
        candidates = list(spawn_points)
        random.shuffle(candidates)
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
        best = None
        best_score = -1e9
        t0 = time.time()
        scanned = 0
        for sp in candidates:
            scanned += 1
            if time.time() - t0 > max_search_sec:
                print(f"Highway spawn search timeout after {scanned} candidates.")
                break
            if scanned % 20 == 0:
                print(f"  spawn search progress: {scanned}/{len(candidates)}")
            wp = carla_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None or wp.is_junction:
                continue
            lane_count = self._count_adjacent_driving_lanes(wp)
            straight_score, span = self._straightness_score(wp)
            # Prefer multi-lane and long straight segments (highway-like)
            score = lane_count * 12.0 + span * 3.0 + straight_score
            if score > best_score:
                best_score = score
                best = (sp, lane_count, span, straight_score)
        print(f"Highway spawn search done: scanned={scanned}, best_score={best_score:.2f}")
        return best, best_score

    def connect(self, host='localhost', port=2000):
        print(f"Connecting to CARLA at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        print("Connected to CARLA")

    def load_map(self, map_name):
        """Load a specific map."""
        current_map = self.world.get_map().name
        if map_name not in current_map:
            print(f"Loading map: {map_name}...")
            self.world = self.client.load_world(map_name)
            print("Map load requested. Waiting for world tick...")
            self.world.wait_for_tick(20.0)
            time.sleep(1.0)
            print(f"Map loaded: {map_name}")
        else:
            print(f"Already on map: {current_map}")

    def enable_synchronous_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)
        tm = self.client.get_trafficmanager(self.tm_port)
        tm.set_synchronous_mode(True)
        print("Synchronous mode enabled.")

    def spawn_npcs(self, num_vehicles=NPC_VEHICLE_COUNT, num_walkers=NPC_WALKER_COUNT):
        """Spawn traffic vehicles and pedestrians."""
        bp_lib = self.world.get_blueprint_library()
        traffic_manager = self.client.get_trafficmanager(self.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.global_percentage_speed_difference(0.0)

        # Spawn NPC vehicles with autopilot.
        vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*") if bp.has_attribute("number_of_wheels")]
        vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute("number_of_wheels")) == 4]
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        vehicle_count = min(num_vehicles, len(spawn_points))
        vehicle_batch = []
        for i in range(vehicle_count):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "autopilot")
            transform = spawn_points[i]
            vehicle_batch.append(
                carla.command.SpawnActor(bp, transform).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port)
                )
            )
        vehicle_results = self.client.apply_batch_sync(vehicle_batch, True)
        for res in vehicle_results:
            if not res.error:
                self.npc_vehicle_ids.append(res.actor_id)
        print(f"Spawned NPC vehicles: {len(self.npc_vehicle_ids)}/{num_vehicles}")

        # Spawn walkers.
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walker_spawn_points = []
        attempts = 0
        max_attempts = max(num_walkers * 5, 100)
        while len(walker_spawn_points) < num_walkers and attempts < max_attempts:
            loc = self.world.get_random_location_from_navigation()
            attempts += 1
            if loc is None:
                continue
            walker_spawn_points.append(carla.Transform(loc))

        walker_batch = []
        walker_speeds = []
        for transform in walker_spawn_points:
            bp = random.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            speed = 1.4
            if bp.has_attribute("speed"):
                speed_values = bp.get_attribute("speed").recommended_values
                if len(speed_values) > 1:
                    speed = float(speed_values[1])
            walker_speeds.append(speed)
            walker_batch.append(carla.command.SpawnActor(bp, transform))

        walker_results = self.client.apply_batch_sync(walker_batch, True)
        spawned_walker_ids = []
        spawned_walker_speeds = []
        for idx, res in enumerate(walker_results):
            if not res.error:
                spawned_walker_ids.append(res.actor_id)
                spawned_walker_speeds.append(walker_speeds[idx])
        self.npc_walker_ids = spawned_walker_ids

        # Spawn walker AI controllers and assign random destinations.
        walker_controller_bp = bp_lib.find("controller.ai.walker")
        controller_batch = [
            carla.command.SpawnActor(walker_controller_bp, carla.Transform(), wid)
            for wid in self.npc_walker_ids
        ]
        controller_results = self.client.apply_batch_sync(controller_batch, True)
        self.npc_walker_controller_ids = [res.actor_id for res in controller_results if not res.error]
        controller_actors = self.world.get_actors(self.npc_walker_controller_ids)

        for i, controller in enumerate(controller_actors):
            controller.start()
            dest = self.world.get_random_location_from_navigation()
            if dest is not None:
                controller.go_to_location(dest)
            controller.set_max_speed(float(spawned_walker_speeds[i] if i < len(spawned_walker_speeds) else 1.4))

        print(f"Spawned NPC walkers: {len(self.npc_walker_ids)}/{num_walkers}")

    def spawn_ego_vehicle(self):
        print("Selecting spawn point...")
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('role_name', 'hero')
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        self.spawn_meta = {"strategy": "random_spawn"}
        print("Selected random spawn point.")
        print("Spawning ego vehicle...")
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned ego vehicle at {spawn_point.location}")
        return self.ego_vehicle

    def setup_cameras(self):
        print("Setting up cameras...")
        bp_lib = self.world.get_blueprint_library()
        for name, cfg in self.camera_configs.items():
            print(f"  - spawning {name}")
            cam_bp = bp_lib.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(IMG_WIDTH))
            cam_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
            cam_bp.set_attribute('fov', str(cfg['fov']))
            cam_bp.set_attribute('enable_postprocess_effects', 'True')
            cam_bp.set_attribute('sensor_tick', '0.0')

            transform = carla.Transform(
                carla.Location(x=cfg['x'], y=cfg['y'], z=cfg['z']),
                carla.Rotation(pitch=cfg['pitch'], yaw=cfg['yaw'])
            )
            sensor = self.world.spawn_actor(cam_bp, transform, attach_to=self.ego_vehicle)
            self.sensor_queues[name] = queue.Queue()
            sensor.listen(lambda data, n=name: self._camera_callback(data, n))
            self.sensors[name] = sensor
            time.sleep(0.2)

        print(f"Setup {len(self.sensors)} cameras ({IMG_WIDTH}x{IMG_HEIGHT})")

    def _camera_callback(self, image, name):
        if not self.sensor_queues[name].full():
            self.sensor_queues[name].put(image)

    def get_camera_images(self):
        images = []
        for name in self.camera_order:
            try:
                data = self.sensor_queues[name].get(timeout=1.0)
                array = np.frombuffer(data.raw_data, dtype=np.uint8)
                array = array.reshape((IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3]
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                images.append(array)
            except queue.Empty:
                images.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8))
        return np.array(images)

    def get_ego_state(self):
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        return {
            'x': transform.location.x,
            'y': transform.location.y,
            'z': transform.location.z,
            'roll': transform.rotation.roll,
            'pitch': transform.rotation.pitch,
            'yaw': transform.rotation.yaw,
            'speed': math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
        }

    def update_history(self, state):
        self.history_buffer.append(state)
        if len(self.history_buffer) > NUM_HISTORY:
            self.history_buffer.pop(0)

    def get_history_in_local_frame(self):
        if len(self.history_buffer) < NUM_HISTORY:
            while len(self.history_buffer) < NUM_HISTORY:
                self.history_buffer.insert(0, self.history_buffer[0] if self.history_buffer else self.get_ego_state())

        current = self.history_buffer[-1]
        current_pos = np.array([current['x'], current['y'], current['z']])
        yaw_rad = math.radians(-current['yaw'])
        cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
        rot_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        history_xyz = np.zeros((NUM_HISTORY, 3), dtype=np.float32)
        history_rot = np.zeros((NUM_HISTORY, 3, 3), dtype=np.float32)

        for i, state in enumerate(self.history_buffer):
            pos = np.array([state['x'], state['y'], state['z']])
            history_xyz[i] = rot_matrix @ (pos - current_pos)
            state_yaw = math.radians(-state['yaw'])
            rel_yaw = state_yaw - yaw_rad
            history_rot[i] = np.array([
                [math.cos(rel_yaw), -math.sin(rel_yaw), 0],
                [math.sin(rel_yaw), math.cos(rel_yaw), 0],
                [0, 0, 1]
            ])

        return history_xyz, history_rot

    def apply_control(self, steering, throttle, brake):
        control = carla.VehicleControl()
        control.steer = float(steering)
        control.throttle = float(throttle)
        control.brake = float(brake)
        self.ego_vehicle.apply_control(control)

    def tick(self):
        self.world.tick()

    def cleanup(self):
        print("\nCleaning up...")
        if self.client is None or self.world is None:
            return
        try:
            self.client.get_trafficmanager(self.tm_port).set_synchronous_mode(False)
        except Exception:
            pass
        if self.npc_walker_controller_ids:
            try:
                controllers = self.world.get_actors(self.npc_walker_controller_ids)
                for controller in controllers:
                    try:
                        controller.stop()
                    except Exception:
                        pass
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_walker_controller_ids])
            except Exception:
                pass
        if self.npc_walker_ids:
            try:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_walker_ids])
            except Exception:
                pass
        if self.npc_vehicle_ids:
            try:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_vehicle_ids])
            except Exception:
                pass
        for sensor in self.sensors.values():
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:
                pass
        if self.ego_vehicle:
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except Exception:
            pass


# ============================================================================
# Local Inference
# ============================================================================
def prepare_model_input(images_array, history_xyz, history_rot):
    """Convert CARLA data to model input format."""
    # images_array: (NUM_CAMERAS, NUM_FRAMES, H, W, C) uint8
    images = torch.from_numpy(images_array)
    images = rearrange(images, "n t h w c -> n t c h w")

    hist_xyz = torch.from_numpy(history_xyz).float().unsqueeze(0).unsqueeze(0)
    hist_rot = torch.from_numpy(history_rot).float().unsqueeze(0).unsqueeze(0)

    return {
        'image_frames': images,
        'ego_history_xyz': hist_xyz,
        'ego_history_rot': hist_rot,
    }


def run_inference(model, processor, data):
    """Run Alpamayo inference locally."""
    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    torch.cuda.manual_seed_all(42)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=copy.deepcopy(model_inputs),
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=NUM_TRAJ_SAMPLES,
            diffusion_kwargs={"inference_step": 10},
            max_generation_length=256,
            return_extra=True,
        )

    return pred_xyz, extra


def extract_cot_text(extra):
    """Safely extract a single CoT string from model extra output."""
    if not isinstance(extra, dict):
        return ""
    if "cot" not in extra:
        return ""

    cot = extra["cot"]
    if cot is None:
        return ""

    # Unwrap nested containers (list/tuple/ndarray/tensor-like) until scalar-like.
    while True:
        if isinstance(cot, str):
            return cot
        if isinstance(cot, (list, tuple)):
            if len(cot) == 0:
                return ""
            cot = cot[0]
            continue
        if isinstance(cot, np.ndarray):
            if cot.size == 0:
                return ""
            cot = cot.flat[0]
            continue
        if hasattr(cot, "numel") and hasattr(cot, "reshape"):
            # torch tensor path
            if int(cot.numel()) == 0:
                return ""
            cot = cot.reshape(-1)[0].item()
            continue
        return str(cot)


def extract_trajectory_samples(pred_xyz):
    """Extract trajectory samples as (num_samples, horizon, 3) from model output tensor."""
    arr = pred_xyz.detach().cpu().numpy()
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected pred_xyz shape after squeeze: {arr.shape}")
    if arr.shape[-1] < 3:
        raise ValueError(f"Trajectory last dim must be >= 3, got {arr.shape}")
    return arr[:, :, :3]


def select_trajectory_by_prev_similarity(traj_samples, prev_traj):
    """Choose trajectory closest to previous selected trajectory in XY space."""
    num_samples = traj_samples.shape[0]
    if prev_traj is None:
        return 0, [None] * num_samples

    prev_xy = prev_traj[:, :2]
    scores = []
    for i in range(num_samples):
        curr_xy = traj_samples[i, :, :2]
        n = min(len(curr_xy), len(prev_xy))
        if n <= 0:
            score = float("inf")
        else:
            score = float(np.mean(np.linalg.norm(curr_xy[:n] - prev_xy[:n], axis=1)))
        scores.append(score)

    best_idx = int(np.argmin(scores))
    return best_idx, scores


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    use_quantization = args.quantization

    print("=" * 60)
    print("CARLA Real-time Control with Alpamayo (Local)")
    print("=" * 60)
    print(f"Quantization: {'ON (4-bit)' if use_quantization else 'OFF (full-precision)'}")

    # Load model
    print("\nLoading model...")
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:
        model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            dtype=torch.bfloat16
        ).to("cuda")

    processor = helper.get_processor(model.tokenizer)
    print("Model loaded!")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB allocated")

    # Initialize CARLA
    carla_if = CARLAInterface()

    # Initialize video recorder
    video_recorder = VideoRecorder(OUTPUT_VIDEO, fps=VIDEO_FPS) if SAVE_VIDEO else None

    try:
        carla_if.connect()
        carla_if.load_map(CARLA_MAP)
        carla_if.spawn_ego_vehicle()
        carla_if.enable_synchronous_mode()
        carla_if.spawn_npcs(num_vehicles=NPC_VEHICLE_COUNT, num_walkers=NPC_WALKER_COUNT)
        carla_if.setup_cameras()
        time.sleep(1.0)

        # Controller
        pid_follower = OfficialPIDFollower(carla_if.world, carla_if.ego_vehicle)

        # State
        current_trajectory = None
        current_pred_xyz = None
        prev_selected_trajectory = None
        current_selected_traj_idx = 0
        current_cot = ""
        current_inference_time = 0.0
        frame_buffer = []
        prev_control = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}

        print("\nStarting control loop...")
        if SAVE_VIDEO:
            print(f"Recording video to: {OUTPUT_VIDEO}")
        print("-" * 60)

        frame_count = 0

        while True:
            carla_if.tick()
            frame_count += 1

            state = carla_if.get_ego_state()
            carla_if.update_history(state)

            images = carla_if.get_camera_images()
            frame_buffer.append(images)
            if len(frame_buffer) > NUM_FRAMES:
                frame_buffer.pop(0)

            # Run inference when we have enough frames
            if len(frame_buffer) >= NUM_FRAMES:
                # Prepare images: (NUM_CAMERAS, NUM_FRAMES, H, W, C)
                images_array = np.zeros(
                    (NUM_CAMERAS, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                    dtype=np.uint8
                )
                for t, frame_images in enumerate(frame_buffer):
                    for c in range(NUM_CAMERAS):
                        images_array[c, t] = frame_images[c]

                history_xyz, history_rot = carla_if.get_history_in_local_frame()

                # Measure pure model inference time only.
                model_data = prepare_model_input(images_array, history_xyz, history_rot)
                model_start_time = time.time()
                pred_xyz, extra = run_inference(model, processor, model_data)
                model_inference_time = time.time() - model_start_time

                # Extract sampled trajectories and select one similar to previous frame.
                traj_samples = extract_trajectory_samples(pred_xyz)
                selected_idx, similarity_scores = select_trajectory_by_prev_similarity(
                    traj_samples, prev_selected_trajectory
                )
                current_selected_traj_idx = selected_idx
                current_trajectory = traj_samples[selected_idx]
                prev_selected_trajectory = current_trajectory.copy()
                current_pred_xyz = traj_samples
                current_cot = extract_cot_text(extra)
                current_inference_time = model_inference_time

                print(f"[Frame {frame_count}] Inference: {model_inference_time:.2f}s")
                print(f"    CoT: {current_cot[:60]}...")
                print(f"    Selected traj sample: {current_selected_traj_idx}/{NUM_TRAJ_SAMPLES - 1}")
                # Debug: print first few trajectory points
                print(f"    Traj[0:3]: {current_trajectory[:3, :2]}")

            # Apply control
            if current_trajectory is not None:
                vehicle_tf = carla_if.ego_vehicle.get_transform()
                steering_raw, throttle_raw, brake_raw, ctrl_debug = pid_follower.compute_control(
                    vehicle_tf, current_trajectory[:, :3], float(state["speed"])
                )

                # Smooth control to avoid abrupt command jumps.
                alpha = CONTROL_SMOOTH_ALPHA
                steering = (1.0 - alpha) * prev_control["steer"] + alpha * steering_raw
                throttle = (1.0 - alpha) * prev_control["throttle"] + alpha * throttle_raw
                brake = (1.0 - alpha) * prev_control["brake"] + alpha * brake_raw

                # Do not apply throttle and brake simultaneously.
                if throttle >= brake:
                    brake = 0.0
                else:
                    throttle = 0.0

                prev_control = {"steer": steering, "throttle": throttle, "brake": brake}
                carla_if.apply_control(steering, throttle, brake)

                # Record visualization frame
                if SAVE_VIDEO and current_pred_xyz is not None:
                    # Use cam_front_wide (index 1) for visualization
                    cam_img = images[1]  # cam_front_wide
                    vis_frame = create_visualization_frame(
                        cam_img, current_pred_xyz, current_selected_traj_idx, frame_count,
                        current_inference_time, current_cot,
                        state['speed'] * 3.6, steering
                    )
                    video_recorder.add_frame(vis_frame)

                print(f"[Frame {frame_count}] Speed: {state['speed']*3.6:.1f} km/h, "
                    f"Steer: {steering:.4f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}")
            else:
                # Before first inference: slowly move forward to build history
                carla_if.apply_control(0.0, 0.3, 0.0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        # Save video before cleanup
        if SAVE_VIDEO and video_recorder:
            video_recorder.save()
        carla_if.cleanup()

    print("\nStopped.")


if __name__ == "__main__":
    main()
