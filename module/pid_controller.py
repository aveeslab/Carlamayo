"""PID controller helpers and official CARLA follower."""

import math
import os
import sys

import carla
import numpy as np

from . import config as cfg


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
            candidate_roots.append(os.path.expanduser(v))

    candidate_roots.append(os.path.expanduser(cfg.CARLA_AGENT_ROOT))

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
            max_steering=0.8,
        )

    def _pick_target(self, wp_world, speed_mps):
        lookahead_m = float(
            np.clip(
                cfg.PID_LOOKAHEAD_MIN_M + cfg.PID_LOOKAHEAD_SPEED_GAIN * speed_mps,
                cfg.PID_LOOKAHEAD_MIN_M,
                cfg.PID_LOOKAHEAD_MAX_M,
            )
        )
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
        target_speed_kmh = float(
            np.clip(
                cfg.PID_TARGET_SPEED_MIN_KMH + cfg.PID_TARGET_SPEED_EXTENT_GAIN * traj_extent,
                cfg.PID_TARGET_SPEED_MIN_KMH,
                cfg.PID_TARGET_SPEED_MAX_KMH,
            )
        )
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
