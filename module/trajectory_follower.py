"""Follow an Alpamayo trajectory in CARLA: pure-pursuit steering, feedforward + PI speed.

Alpamayo replans every tick with sampled CoT and diffusion noise, so the plan the
controller sees is noisy and restarts from the current pose each time. The follower
therefore tracks the plan in the world frame from the waypoint nearest the rear
axle (steering) and from the plan's own timing (speed), filters one-tick dips in
the planned speed, and holds planned stops. Speed control is a measured
throttle feedforward plus a PI with anti-windup, not CARLA's ``agents`` PID, whose
integral only spans one second and whose km/h gains saturate at ~1 km/h error.
"""

from __future__ import annotations

import math
from collections import deque

import carla
import numpy as np

from . import config as cfg

# Alpamayo predicts 64 waypoints at 10 Hz (6.4 s).
WAYPOINT_DT = 0.1


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


def pure_pursuit_steer(ego_xy, yaw_rad, target_xy, wheelbase_m, max_steer_rad):
    """Normalized CARLA steer [-1, 1] that drives the rear axle onto ``target_xy``.

    CARLA is left-handed (x forward, y right, yaw clockwise from above), so a target to
    the right gives a positive bearing and a positive (right) steer.
    """
    dx, dy = target_xy[0] - ego_xy[0], target_xy[1] - ego_xy[1]
    forward = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
    right = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
    distance = math.hypot(dx, dy)
    if distance < 1e-3:
        return 0.0
    bearing = math.atan2(right, forward)
    curvature = 2.0 * math.sin(bearing) / distance
    wheel_angle = math.atan(wheelbase_m * curvature)
    return float(np.clip(wheel_angle / max_steer_rad, -1.0, 1.0))


def max_path_curvature(path_xy):
    """Largest curvature (1/m) over consecutive waypoint triples of a 2-D path."""
    pts = np.asarray(path_xy, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    ab, bc, ca = b - a, c - b, a - c
    cross = np.abs(ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0])
    denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ca, axis=1)
    valid = denom > 1e-6
    if not np.any(valid):
        return 0.0
    # Menger curvature: 4 * triangle area / product of side lengths.
    return float(np.max(2.0 * cross[valid] / denom[valid]))


class TrajectoryFollower:
    """Stateful follower for one ego vehicle; create a new one after a respawn."""

    def __init__(self, vehicle):
        wheels = vehicle.get_physics_control().wheels
        self.max_steer_rad = math.radians(max(w.max_steer_angle for w in wheels))
        self._integral = 0.0
        self._prev_speed = None
        self._prev_steer = 0.0
        self._recent_targets = deque(maxlen=cfg.FOLLOW_TARGET_MEDIAN_TICKS)
        self._launching = False
        self._braking = False

    def _target_speed(self, speeds_mps, speed_mps, elapsed_sec):
        """Planned speed PID_SPEED_LEAD_SEC ahead, with the start-from-rest launch rule."""
        now = int(elapsed_sec / WAYPOINT_DT + 1e-6)
        step = now + int(cfg.PID_SPEED_LEAD_SEC / WAYPOINT_DT + 1e-6)
        preview = max(1, round(cfg.PID_SPEED_PREVIEW_SEC / WAYPOINT_DT))
        window = speeds_mps[step:step + preview]
        target = float(np.mean(window)) if len(window) else 0.0
        # The launch floor applies from rest until the ego actually reaches the launch
        # speed, so a plan that keeps "waiting a second, then going" (re-planned every
        # tick, so the wait never ends) creeps at the launch speed instead of cycling
        # between launch and planned stop. Hysteresis on the plan length: keep launching
        # unless the plan collapses below PID_LAUNCH_HOLD_PLAN_M. The remaining plan is
        # measured from now, without the lead, so an older (async) plan is judged on what
        # it still intends rather than on its decayed tail.
        min_plan_m = cfg.PID_LAUNCH_HOLD_PLAN_M if self._launching else cfg.PID_LAUNCH_MIN_PLAN_M
        max_speed = cfg.PID_LAUNCH_SPEED_MPS if self._launching else cfg.PID_REST_SPEED_MPS
        launch = (
            speed_mps < max_speed
            and float(np.sum(speeds_mps[now:])) * WAYPOINT_DT >= min_plan_m
        )
        self._launching = launch
        if launch:
            target = max(target, cfg.PID_LAUNCH_SPEED_MPS)
        return min(target, cfg.PID_TARGET_SPEED_MAX_KMH / 3.6), launch

    def _speed_control(self, target_mps, speed_mps, planned_stop):
        """Feedforward + PI (conditional integration) + derivative on measurement."""
        if planned_stop:
            # Brake while moving, hold full brake once (nearly) stopped, never creep
            # forward. Reset the integrator so the next launch does not inherit a
            # wound-up term.
            self._integral = 0.0
            self._recent_targets.clear()
            self._prev_speed = speed_mps
            return 0.0, (1.0 if speed_mps < 0.15 else cfg.FOLLOW_BRAKE_ON)
        # A single-tick dip in the replanned speed profile is noise: take the median of
        # the last few targets, which drops it without delaying sustained changes.
        self._recent_targets.append(target_mps)
        target = float(np.median(self._recent_targets))
        error = target - speed_mps
        dv = 0.0 if self._prev_speed is None else (speed_mps - self._prev_speed) / cfg.CONTROL_DT
        self._prev_speed = speed_mps
        feedforward = cfg.FOLLOW_FF_THROTTLE_AT_ZERO + cfg.FOLLOW_FF_THROTTLE_PER_MPS * target
        effort = (
            feedforward + cfg.FOLLOW_KP * error + cfg.FOLLOW_KI * self._integral
            - cfg.FOLLOW_KD * dv
        )
        saturated = (effort >= cfg.THROTTLE_MAX and error > 0) or (
            effort <= -cfg.BRAKE_MAX and error < 0
        )
        if not saturated:
            self._integral += error * cfg.CONTROL_DT
            limit = cfg.FOLLOW_KI_EFFORT_LIMIT / max(cfg.FOLLOW_KI, 1e-9)
            self._integral = float(np.clip(self._integral, -limit, limit))
        # Measured MKZ: any brake value from 0.03 up brakes ~6.9 m/s^2 (on/off), while
        # pedals released shed about 0.4 * speed m/s^2. Coasting therefore covers a target
        # down to ~60% of the current speed within the one-second lead; brake below that,
        # with hysteresis so the on/off brake does not chatter around the threshold.
        ratio = cfg.FOLLOW_BRAKE_RELEASE_RATIO if self._braking else cfg.FOLLOW_BRAKE_SPEED_RATIO
        self._braking = effort < 0.0 and target < ratio * speed_mps
        if self._braking:
            return 0.0, cfg.FOLLOW_BRAKE_ON
        return min(max(effort, 0.0), cfg.THROTTLE_MAX), 0.0

    def compute_control(self, vehicle_tf, wp_world, speed_mps, speeds_mps, elapsed_sec=0.0):
        """Return ``(steer, throttle, brake, debug)`` for a world-frame plan.

        ``wp_world`` comes from ``trajectory_to_world`` for the pose the plan was
        predicted at; ``speeds_mps`` from ``planned_speeds``; ``elapsed_sec`` is the
        simulation time since that prediction (non-zero with --async).
        """
        rear_axle = vehicle_tf.transform(carla.Location(x=cfg.EGO_REAR_AXLE_X))
        ego_xy = np.array([rear_axle.x, rear_axle.y])
        yaw = math.radians(vehicle_tf.rotation.yaw)
        start_idx = int(np.argmin(np.linalg.norm(wp_world[:, :2] - ego_xy, axis=1)))
        path = wp_world[start_idx:, :2]

        lookahead_m = float(np.clip(
            cfg.PID_LOOKAHEAD_MIN_M + cfg.PID_LOOKAHEAD_SPEED_GAIN * speed_mps,
            cfg.PID_LOOKAHEAD_MIN_M, cfg.PID_LOOKAHEAD_MAX_M,
        ))
        distances = np.linalg.norm(path - ego_xy, axis=1)
        ahead = np.nonzero(distances >= lookahead_m)[0]
        target_idx = int(ahead[0]) if len(ahead) else len(path) - 1

        target_mps, launch = self._target_speed(speeds_mps, speed_mps, elapsed_sec)
        # Vehicle limit the plan may ignore: cap speed by the path curvature ahead so the
        # lateral acceleration stays below FOLLOW_LAT_ACCEL_MAX (a plan drawn at 35 km/h
        # through a tight bend is otherwise driven straight into the outside).
        curvature = max_path_curvature(path[: target_idx + 1])
        curve_limit = math.inf
        if curvature > 1e-6:
            curve_limit = math.sqrt(cfg.FOLLOW_LAT_ACCEL_MAX / curvature)
        target_mps = min(target_mps, max(curve_limit, cfg.PID_LAUNCH_SPEED_MPS))
        planned_stop = target_mps <= cfg.PID_STOP_SPEED_MPS
        throttle, brake = self._speed_control(target_mps, speed_mps, planned_stop)
        target = float(np.median(self._recent_targets)) if self._recent_targets else 0.0
        steer = pure_pursuit_steer(
            ego_xy, yaw, path[target_idx], cfg.EGO_WHEELBASE_M, self.max_steer_rad
        )
        max_delta = cfg.FOLLOW_STEER_RATE_PER_SEC * cfg.CONTROL_DT
        steer = float(np.clip(steer, self._prev_steer - max_delta, self._prev_steer + max_delta))
        self._prev_steer = steer

        debug = {
            "mode": "pure_pursuit_ff_pi",
            "target_speed_kmh": target_mps * 3.6,
            "target_speed_filtered_kmh": target * 3.6 if not planned_stop else 0.0,
            "lookahead_m": lookahead_m,
            "target_idx": start_idx + target_idx,
            "target_wp_xy": [float(path[target_idx][0]), float(path[target_idx][1])],
            "planned_stop": planned_stop,
            "launch": launch,
            "integral_effort": cfg.FOLLOW_KI * self._integral,
            "curve_limit_kmh": curve_limit * 3.6,
        }
        return steer, float(throttle), float(brake), debug
