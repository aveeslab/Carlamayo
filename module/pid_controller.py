"""PID controller helpers and official CARLA follower."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import carla
import numpy as np

from . import config as cfg

# Alpamayo predicts 64 waypoints at 10 Hz (6.4 s).
WAYPOINT_DT = 0.1


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


def trajectory_to_world(vehicle_tf, wp_ego):
    """World xyz of rear-axle-relative Alpamayo waypoints for the pose they were predicted at."""
    wp_local = alpamayo_to_carla_local(wp_ego)
    # Alpamayo waypoints start at the rear axle; CARLA transforms use the actor origin.
    return local_to_world(vehicle_tf, wp_local + [cfg.EGO_REAR_AXLE_X, 0.0, 0.0])


def planned_speeds(wp_ego):
    """Planned speed (m/s) over each 0.1 s step of a rear-axle-relative Alpamayo trajectory."""
    xy = np.vstack([np.zeros((1, 2)), np.asarray(wp_ego)[:, :2]])
    return np.linalg.norm(np.diff(xy, axis=0), axis=1) / WAYPOINT_DT


@dataclass(frozen=True)
class RawTargetWaypoint:
    """Minimal waypoint-like target for CARLA VehiclePIDController."""

    transform: carla.Transform


class OfficialPIDFollower:
    """CARLA official PID follower for raw Alpamayo targets."""

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
            max_steering=0.8,
        )
        self._launching = False

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
        target_wp = RawTargetWaypoint(carla.Transform(loc, carla.Rotation()))
        return target_wp, target_idx, lookahead_m

    def compute_control(self, vehicle_tf, wp_world, speed_mps, speeds_mps, elapsed_sec=0.0):
        """Follow a world-frame trajectory (``trajectory_to_world``) from the current pose.

        Steering tracks a lookahead point starting at the waypoint nearest the rear
        axle, so the part of an older (async) trajectory the ego has already driven past
        is skipped. Speed follows the plan's own timing: the mean of ``speeds_mps``
        (``planned_speeds``) over the next ``PID_SPEED_PREVIEW_SEC`` after
        ``elapsed_sec`` since the prediction. A planned stop holds the brake.

        From standstill Alpamayo plans a very slow start and every replan restarts it,
        so the ego would never move by tracking the plan alone. While at rest, a plan
        that still covers at least ``PID_LAUNCH_MIN_PLAN_M`` (``PID_LAUNCH_HOLD_PLAN_M``
        once launching) launches the ego at ``PID_LAUNCH_SPEED_MPS``; once moving, the
        plan's own speed takes over.
        """
        rear_axle = vehicle_tf.transform(carla.Location(x=cfg.EGO_REAR_AXLE_X))
        ego_xy = np.array([rear_axle.x, rear_axle.y])
        start_idx = int(np.argmin(np.linalg.norm(wp_world[:, :2] - ego_xy, axis=1)))
        wp_world = wp_world[start_idx:]
        step = int(elapsed_sec / WAYPOINT_DT + 1e-6)
        preview = max(1, round(cfg.PID_SPEED_PREVIEW_SEC / WAYPOINT_DT))
        window = speeds_mps[step:step + preview]
        target_speed_mps = float(np.mean(window)) if len(window) else 0.0
        # Hysteresis: once launching, keep going until moving unless the plan collapses
        # below PID_LAUNCH_HOLD_PLAN_M, so a plan hovering around the threshold cannot stall.
        min_plan_m = cfg.PID_LAUNCH_HOLD_PLAN_M if self._launching else cfg.PID_LAUNCH_MIN_PLAN_M
        launch = (
            speed_mps < cfg.PID_REST_SPEED_MPS
            and float(np.sum(speeds_mps[step:])) * WAYPOINT_DT >= min_plan_m
        )
        self._launching = launch
        if launch:
            target_speed_mps = max(target_speed_mps, cfg.PID_LAUNCH_SPEED_MPS)
        target_speed_kmh = min(target_speed_mps * 3.6, cfg.PID_TARGET_SPEED_MAX_KMH)
        target_wp, target_idx, lookahead_m = self._pick_target(wp_world, speed_mps)
        target_idx += start_idx
        control = self.pid.run_step(target_speed_kmh, target_wp)
        throttle, brake = float(control.throttle), float(control.brake)
        planned_stop = target_speed_mps <= cfg.PID_STOP_SPEED_MPS
        if planned_stop:
            # Same stop handling as CARLA's Alpamayo example: brake at least 0.25 while
            # moving and hold full brake once (nearly) stopped, never creep forward.
            throttle, brake = 0.0, (1.0 if speed_mps < 0.15 else max(brake, 0.25))
        debug = {
            "mode": "official_pid",
            "target_speed_kmh": target_speed_kmh,
            "lookahead_m": lookahead_m,
            "target_idx": int(target_idx),
            "target_wp_xy": [
                float(target_wp.transform.location.x),
                float(target_wp.transform.location.y),
            ],
            "target_raw_xy": [
                float(target_wp.transform.location.x),
                float(target_wp.transform.location.y),
            ],
            "target_projected_to_road": False,
            "planned_stop": planned_stop,
            "launch": launch,
        }
        return float(control.steer), throttle, brake, debug
