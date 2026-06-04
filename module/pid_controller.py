"""PID controller helpers and trajectory-based CARLA follower."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import carla
import numpy as np

from . import config as cfg


def _resolve_vehicle_pid_controller():
    """Import VehiclePIDController, preferring the configured CARLA 0.10 root."""

    candidate_roots = []
    carla_010_root = os.environ.get("CARLA_010_ROOT")
    if carla_010_root:
        candidate_roots.append(os.path.expanduser(carla_010_root))
    candidate_roots.append(os.path.expanduser(cfg.CARLA_AGENT_ROOT))

    for root in candidate_roots:
        agents_parent = os.path.abspath(os.path.join(root, "PythonAPI", "carla"))
        if os.path.isdir(agents_parent) and agents_parent not in sys.path:
            sys.path.insert(0, agents_parent)

    try:
        from agents.navigation.controller import VehiclePIDController as _VehiclePIDController

        return _VehiclePIDController
    except ImportError as e:
        raise ImportError(
            "VehiclePIDController not found. Set CARLA_010_ROOT or add "
            "'<CARLA_010_ROOT>/PythonAPI/carla' to PYTHONPATH."
        ) from e


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


@dataclass(frozen=True)
class RawTargetWaypoint:
    """Minimal waypoint-like target for CARLA VehiclePIDController."""

    transform: carla.Transform


@dataclass(frozen=True)
class LookaheadTarget:
    """Interpolated trajectory lookahead target."""

    local_xyz: np.ndarray
    target_idx: int
    lookahead_path_m: float
    trajectory_path_length_m: float
    segment_idx: int
    segment_ratio: float


class OfficialPIDFollower:
    """CARLA follower using official longitudinal PID and trajectory pure-pursuit steering."""

    def __init__(self, world, vehicle):
        VehiclePIDController = _resolve_vehicle_pid_controller()
        self.world = world
        self.vehicle = vehicle
        args_lateral = {
            "K_P": cfg.PID_LAT_KP,
            "K_I": cfg.PID_LAT_KI,
            "K_D": cfg.PID_LAT_KD,
            "dt": cfg.CONTROL_DT,
        }
        args_longitudinal = {
            "K_P": cfg.PID_LON_KP,
            "K_I": cfg.PID_LON_KI,
            "K_D": cfg.PID_LON_KD,
            "dt": cfg.CONTROL_DT,
        }
        self.pid = VehiclePIDController(
            vehicle,
            args_lateral=args_lateral,
            args_longitudinal=args_longitudinal,
            max_throttle=cfg.THROTTLE_MAX,
            max_brake=cfg.BRAKE_MAX,
            max_steering=cfg.PID_MAX_STEER,
        )

    def _lookahead_m(self, speed_mps):
        return float(
            np.clip(
                cfg.PID_LOOKAHEAD_MIN_M + cfg.PID_LOOKAHEAD_SPEED_GAIN * speed_mps,
                cfg.PID_LOOKAHEAD_MIN_M,
                cfg.PID_LOOKAHEAD_MAX_M,
            )
        )

    def _interpolate_lookahead_target(self, wp_local, lookahead_m):
        if len(wp_local) == 0:
            return None

        path = np.vstack([np.zeros((1, 3), dtype=np.float64), wp_local[:, :3]])
        segment_vectors = np.diff(path[:, :2], axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        trajectory_path_length_m = float(np.sum(segment_lengths))
        if trajectory_path_length_m <= 1e-6:
            return LookaheadTarget(
                local_xyz=path[-1].copy(),
                target_idx=max(0, len(wp_local) - 1),
                lookahead_path_m=0.0,
                trajectory_path_length_m=trajectory_path_length_m,
                segment_idx=0,
                segment_ratio=0.0,
            )

        target_s = float(np.clip(lookahead_m, 0.0, trajectory_path_length_m))
        distance_before = 0.0
        last_valid_idx = 0
        for segment_idx, segment_length in enumerate(segment_lengths):
            if segment_length <= 1e-6:
                continue
            last_valid_idx = segment_idx
            distance_after = distance_before + float(segment_length)
            if target_s <= distance_after:
                segment_ratio = (target_s - distance_before) / float(segment_length)
                local_xyz = path[segment_idx] + segment_ratio * (path[segment_idx + 1] - path[segment_idx])
                return LookaheadTarget(
                    local_xyz=local_xyz,
                    target_idx=min(segment_idx, len(wp_local) - 1),
                    lookahead_path_m=target_s,
                    trajectory_path_length_m=trajectory_path_length_m,
                    segment_idx=segment_idx,
                    segment_ratio=float(segment_ratio),
                )
            distance_before = distance_after

        return LookaheadTarget(
            local_xyz=path[-1].copy(),
            target_idx=len(wp_local) - 1,
            lookahead_path_m=trajectory_path_length_m,
            trajectory_path_length_m=trajectory_path_length_m,
            segment_idx=last_valid_idx,
            segment_ratio=1.0,
        )

    def _target_waypoint(self, vehicle_tf, target_local_xyz):
        target_world = local_to_world(vehicle_tf, target_local_xyz[None, :])[0]
        loc = carla.Location(
            x=float(target_world[0]),
            y=float(target_world[1]),
            z=float(target_world[2]),
        )
        return RawTargetWaypoint(carla.Transform(loc, carla.Rotation()))

    def _pure_pursuit_steer(self, target_local_xyz):
        x = float(target_local_xyz[0])
        y = float(target_local_xyz[1])
        distance_sq = x * x + y * y
        if distance_sq <= 1e-6 or x <= 0.0:
            return 0.0, 0.0, 0.0

        curvature = 2.0 * y / distance_sq
        steer_angle_rad = float(np.arctan(cfg.PID_WHEELBASE_M * curvature))
        steer = float(
            np.clip(
                steer_angle_rad / cfg.PID_STEER_NORMALIZATION_RAD,
                -cfg.PID_MAX_STEER,
                cfg.PID_MAX_STEER,
            )
        )
        return steer, float(curvature), steer_angle_rad

    def compute_control(self, vehicle_tf, wp_ego, speed_mps):
        wp_local = alpamayo_to_carla_local(wp_ego)
        traj_extent = float(np.max(np.linalg.norm(wp_local[:, :2], axis=1))) if len(wp_local) else 0.0
        target_speed_kmh = float(
            np.clip(
                cfg.PID_TARGET_SPEED_MIN_KMH + cfg.PID_TARGET_SPEED_EXTENT_GAIN * traj_extent,
                cfg.PID_TARGET_SPEED_MIN_KMH,
                cfg.PID_TARGET_SPEED_MAX_KMH,
            )
        )

        lookahead_m = self._lookahead_m(speed_mps)
        target = self._interpolate_lookahead_target(wp_local, lookahead_m)
        if target is None:
            return 0.0, 0.0, 0.0, {
                "mode": "trajectory_pure_pursuit_no_target",
                "traj_extent": traj_extent,
            }

        target_wp = self._target_waypoint(vehicle_tf, target.local_xyz)
        longitudinal_control = self.pid.run_step(target_speed_kmh, target_wp)
        steer, curvature, steer_angle_rad = self._pure_pursuit_steer(target.local_xyz)

        debug = {
            "mode": "trajectory_pure_pursuit",
            "target_speed_kmh": target_speed_kmh,
            "lookahead_m": lookahead_m,
            "lookahead_path_m": target.lookahead_path_m,
            "trajectory_path_length_m": target.trajectory_path_length_m,
            "target_idx": int(target.target_idx),
            "target_segment_idx": int(target.segment_idx),
            "target_segment_ratio": float(target.segment_ratio),
            "target_local_xy": [
                float(target.local_xyz[0]),
                float(target.local_xyz[1]),
            ],
            "target_wp_xy": [
                float(target_wp.transform.location.x),
                float(target_wp.transform.location.y),
            ],
            "target_raw_xy": [
                float(target_wp.transform.location.x),
                float(target_wp.transform.location.y),
            ],
            "target_projected_to_road": False,
            "traj_extent": traj_extent,
            "pure_pursuit_curvature": curvature,
            "pure_pursuit_steer_angle_rad": steer_angle_rad,
        }
        return (
            steer,
            float(longitudinal_control.throttle),
            float(longitudinal_control.brake),
            debug,
        )
