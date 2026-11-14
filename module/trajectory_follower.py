"""Follow an Alpamayo trajectory in CARLA with the official example's tracker.

A port of ``TrajectoryTracker`` from CARLA's official Alpamayo example
(``PythonAPI/examples/av_stacks/alpamayo/alpamayo_controller.py``): pure-pursuit steering
to the first plan point at least ``cfg.TRACKER_LOOKAHEAD_M`` past the point nearest the
rear axle, rate-limited, and a PID on the plan's speed at the current trajectory time
step, giving one signed effort (throttle positive, brake negative). The plan and the
ego pose are both expressed in the CARLA-local (y right) rear-axle frame of the pose
the plan was predicted from.

Two additions. For CARLA 0.9.16, the effort carries a throttle feedforward measured for
its MKZ, which needs ~0.36 throttle to move at all; the official PID alone (tuned on the
UE5 MKZ) leaves it creeping at a few cm/s and unable to pull away from a stop. And a plan
that begins at rest but starts moving within ``cfg.TRACKER_LAUNCH_WINDOW_S`` is followed
from its start, since replanning from rest would otherwise delay that start forever.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config as cfg


def alpamayo_to_carla_local(wp_ego):
    """Convert Alpamayo local frame (y=left) to CARLA local frame (y=right)."""
    wp_local = np.asarray(wp_ego, dtype=np.float64).copy()
    wp_local[:, 1] *= -1.0
    return wp_local


def pose_in_plan_frame(plan_tf, vehicle_tf):
    """Rear-axle position and yaw (rad) of ``vehicle_tf`` in the rear-axle frame of ``plan_tf``.

    Both are ``carla.Transform`` actor poses; the result is CARLA-local (y right).
    """
    rel = np.array(plan_tf.get_inverse_matrix()) @ np.array(vehicle_tf.get_matrix())
    rear_axle = np.array([cfg.EGO_REAR_AXLE_X, 0.0, 0.0])
    position = (rel @ np.append(rear_axle, 1.0))[:3] - rear_axle
    return position, math.atan2(rel[1, 0], rel[0, 0])


@dataclass(frozen=True)
class ControlCommand:
    throttle: float
    brake: float
    steer: float
    target_speed_mps: float
    target_index: int
    requested_steer: float = 0.0
    speed_index: int = 0
    speed_error_mps: float = 0.0
    longitudinal_effort: float = 0.0
    hold_active: bool = False


class TrajectoryTracker:
    """Stateful tracker for one ego vehicle; ``reset()`` after a respawn or a brake hold."""

    def __init__(
        self,
        wheelbase_m=cfg.EGO_WHEELBASE_M,
        max_wheel_angle_deg=cfg.TRACKER_MAX_WHEEL_ANGLE_DEG,
        lookahead_m=cfg.TRACKER_LOOKAHEAD_M,
        speed_kp=cfg.TRACKER_SPEED_KP,
        speed_ki=cfg.TRACKER_SPEED_KI,
        speed_kd=cfg.TRACKER_SPEED_KD,
        speed_integral_limit=cfg.TRACKER_SPEED_INTEGRAL_LIMIT,
        stop_target_speed_mps=cfg.TRACKER_STOP_TARGET_SPEED_MPS,
        hold_speed_mps=cfg.TRACKER_HOLD_SPEED_MPS,
        minimum_stop_brake=cfg.TRACKER_MIN_STOP_BRAKE,
        hold_brake=cfg.TRACKER_HOLD_BRAKE,
        steering_gain=cfg.TRACKER_STEERING_GAIN,
        max_steer_rate_per_s=cfg.TRACKER_MAX_STEER_RATE_PER_S,
        ff_throttle_at_zero=cfg.TRACKER_FF_THROTTLE_AT_ZERO,
        ff_throttle_per_mps=cfg.TRACKER_FF_THROTTLE_PER_MPS,
        launch_window_s=cfg.TRACKER_LAUNCH_WINDOW_S,
    ):
        self.wheelbase_m = wheelbase_m
        self.max_wheel_angle_rad = np.deg2rad(max_wheel_angle_deg)
        self.lookahead_m = lookahead_m
        self.speed_kp = speed_kp
        self.speed_ki = speed_ki
        self.speed_kd = speed_kd
        self.speed_integral_limit = speed_integral_limit
        self.stop_target_speed_mps = stop_target_speed_mps
        self.hold_speed_mps = hold_speed_mps
        self.minimum_stop_brake = minimum_stop_brake
        self.hold_brake = hold_brake
        self.steering_gain = steering_gain
        self.max_steer_rate_per_s = max_steer_rate_per_s
        self.ff_throttle_at_zero = ff_throttle_at_zero
        self.ff_throttle_per_mps = ff_throttle_per_mps
        self.launch_window_s = launch_window_s
        self.reset()

    def reset(self):
        """Reset steering-rate and longitudinal PID state."""
        self._previous_steer = 0.0
        self._speed_integral = 0.0
        self._previous_speed_mps = None

    def _longitudinal_command(self, target_speed_mps, current_speed_mps, time_step):
        """Return throttle, brake, error, signed PID effort, and hold state."""
        stop_requested = target_speed_mps <= self.stop_target_speed_mps
        if stop_requested:
            target_speed_mps = 0.0
            # Do not let positive integral accumulated while cruising oppose a newly
            # requested stop.
            self._speed_integral = min(self._speed_integral, 0.0)

        error = target_speed_mps - current_speed_mps
        feedforward = 0.0 if stop_requested else (
            self.ff_throttle_at_zero + self.ff_throttle_per_mps * target_speed_mps
        )
        speed_derivative = 0.0
        if self._previous_speed_mps is not None:
            # Derivative on measurement avoids a kick when a newly sampled trajectory
            # changes the requested speed at a replan boundary.
            speed_derivative = -(current_speed_mps - self._previous_speed_mps) / time_step
        self._previous_speed_mps = current_speed_mps

        candidate_integral = float(np.clip(
            self._speed_integral + error * time_step,
            -self.speed_integral_limit, self.speed_integral_limit,
        ))
        candidate_effort = (
            feedforward
            + self.speed_kp * error
            + self.speed_ki * candidate_integral
            + self.speed_kd * speed_derivative
        )
        # Conditional integration: do not wind the integral farther into a saturated
        # pedal command.
        if not (
            (candidate_effort > 1.0 and error > 0.0) or (candidate_effort < -1.0 and error < 0.0)
        ):
            self._speed_integral = candidate_integral
        effort = float(np.clip(
            feedforward
            + self.speed_kp * error
            + self.speed_ki * self._speed_integral
            + self.speed_kd * speed_derivative,
            -1.0, 1.0,
        ))

        if stop_requested:
            if current_speed_mps <= self.hold_speed_mps:
                return 0.0, self.hold_brake, error, -self.hold_brake, True
            brake = max(-effort, self.minimum_stop_brake)
            return 0.0, float(np.clip(brake, 0.0, 1.0)), error, -brake, False
        if effort >= 0.0:
            return effort, 0.0, error, effort, False
        return 0.0, -effort, error, effort, False

    def command(
        self, path_carla_local, current_position_local, current_yaw_local_rad,
        current_speed_mps, time_step=cfg.CONTROL_DT, trajectory_time_index=0,
    ):
        path = np.asarray(path_carla_local, dtype=np.float64)
        position = np.asarray(current_position_local, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] < 3 or not len(path):
            raise ValueError("path must have shape [T, >=3]")
        distances = np.linalg.norm(path[:, :2] - position[:2], axis=1)
        nearest = int(np.argmin(distances))
        candidates = np.flatnonzero(distances[nearest:] >= self.lookahead_m)
        target_index = nearest + int(candidates[0]) if len(candidates) else len(path) - 1

        delta = path[target_index, :2] - position[:2]
        cosine, sine = np.cos(current_yaw_local_rad), np.sin(current_yaw_local_rad)
        target_vehicle = np.array(
            [cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1]]
        )
        distance_sq = max(float(target_vehicle @ target_vehicle), 1e-6)
        curvature = 2.0 * target_vehicle[1] / distance_sq
        wheel_angle = np.arctan(self.wheelbase_m * curvature)
        requested_steer = float(np.clip(
            self.steering_gain * wheel_angle / self.max_wheel_angle_rad, -1.0, 1.0
        ))
        max_steer_delta = self.max_steer_rate_per_s * time_step
        steer = float(np.clip(
            requested_steer,
            self._previous_steer - max_steer_delta,
            self._previous_steer + max_steer_delta,
        ))
        self._previous_steer = steer

        # Alpamayo waypoints are sampled at 10 Hz beginning at t0 + 0.1 s. Longitudinal
        # control therefore follows trajectory time, independent of the spatial
        # pure-pursuit point selected above for steering.
        deltas = np.diff(np.vstack([np.zeros((1, 3)), path[:, :3]]), axis=0)
        speeds = np.linalg.norm(deltas[:, :2], axis=1) / time_step
        # A plan beginning at rest (its first step is below the stop speed, which a moving
        # ego cannot reach in 0.1 s) that starts moving soon is followed from that start.
        moving = np.flatnonzero(speeds > self.stop_target_speed_mps)
        if (
            speeds[0] <= self.stop_target_speed_mps
            and len(moving)
            and moving[0] * time_step < self.launch_window_s
        ):
            trajectory_time_index += int(moving[0])
        speed_index = min(max(int(trajectory_time_index), 0), len(speeds) - 1)
        target_speed = float(speeds[speed_index])
        throttle, brake, speed_error, effort, hold_active = self._longitudinal_command(
            target_speed, current_speed_mps, time_step
        )
        if target_speed <= self.stop_target_speed_mps:
            target_speed = 0.0
        return ControlCommand(
            throttle, brake, steer, target_speed, target_index, requested_steer,
            speed_index, speed_error, effort, hold_active,
        )
