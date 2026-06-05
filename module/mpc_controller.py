"""Linear MPC controller for Alpamayo closed-loop trajectory following."""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

from . import config as cfg


def _import_osqp():
    try:
        import osqp  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only on missing runtime dependency
        raise RuntimeError("MPC controller requires the optional 'osqp' package") from exc
    return osqp


@dataclass(frozen=True)
class MPCConfig:
    """Tunable parameters for the kinematic linear MPC follower."""

    dt: float
    horizon: int
    wheelbase_m: float
    max_steer_rad: float
    accel_max: float
    decel_max: float
    w_lat: float
    w_lon: float
    w_heading: float
    w_speed: float
    w_steer: float
    w_accel: float
    w_dsteer: float
    w_daccel: float
    min_speed_kmh: float
    min_speed_forward_min_m: float
    brake_max: float
    reference_horizon_time_s: float
    reference_speed_floor_mps: float
    reference_min_distance_m: float
    reference_max_distance_m: float
    reference_smoothing_window: int
    terminal_slowdown_distance_m: float
    terminal_speed_mps: float
    steering_tau: float = 0.1
    accel_tau: float = 0.1

    @classmethod
    def from_module_config(cls) -> "MPCConfig":
        return cls(
            dt=cfg.MPC_DT,
            horizon=cfg.MPC_HORIZON,
            wheelbase_m=cfg.WHEELBASE_M,
            max_steer_rad=cfg.MAX_STEER_RAD,
            accel_max=cfg.ACCEL_MAX,
            decel_max=cfg.DECEL_MAX,
            w_lat=cfg.W_LAT,
            w_lon=cfg.W_LON,
            w_heading=cfg.W_HEADING,
            w_speed=cfg.W_SPEED,
            w_steer=cfg.W_STEER,
            w_accel=cfg.W_ACCEL,
            w_dsteer=cfg.W_DSTEER,
            w_daccel=cfg.W_DACCEL,
            min_speed_kmh=cfg.MPC_MIN_SPEED_KMH,
            min_speed_forward_min_m=cfg.MPC_MIN_SPEED_FORWARD_MIN_M,
            brake_max=cfg.MPC_BRAKE_MAX,
            reference_horizon_time_s=cfg.MPC_REFERENCE_HORIZON_TIME_S,
            reference_speed_floor_mps=cfg.MPC_REFERENCE_SPEED_FLOOR_MPS,
            reference_min_distance_m=cfg.MPC_REFERENCE_MIN_DISTANCE_M,
            reference_max_distance_m=cfg.MPC_REFERENCE_MAX_DISTANCE_M,
            reference_smoothing_window=cfg.MPC_REFERENCE_SMOOTHING_WINDOW,
            terminal_slowdown_distance_m=cfg.MPC_TERMINAL_SLOWDOWN_DISTANCE_M,
            terminal_speed_mps=cfg.MPC_TERMINAL_SPEED_MPS,
        )

    @classmethod
    def from_json(cls, path: str | Path | None) -> "MPCConfig":
        base = asdict(cls.from_module_config())
        if path is None:
            return cls(**base)
        data = json.loads(Path(path).read_text())
        unknown = sorted(set(data) - set(base))
        if unknown:
            raise ValueError(f"Unknown MPC config keys: {unknown}")
        base.update(data)
        return cls(**base)

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("MPC dt must be positive")
        if self.horizon <= 0:
            raise ValueError("MPC horizon must be positive")
        if self.wheelbase_m <= 0.0:
            raise ValueError("MPC wheelbase must be positive")
        if self.max_steer_rad <= 0.0:
            raise ValueError("MPC max steering angle must be positive")
        if self.accel_max <= 0.0:
            raise ValueError("MPC accel_max must be positive")
        if self.decel_max >= 0.0:
            raise ValueError("MPC decel_max must be negative")
        weights = (
            self.w_lat,
            self.w_lon,
            self.w_heading,
            self.w_speed,
            self.w_steer,
            self.w_accel,
            self.w_dsteer,
            self.w_daccel,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("MPC weights must be non-negative")
        if self.min_speed_kmh < 0.0:
            raise ValueError("MPC min_speed_kmh must be non-negative")
        if self.min_speed_forward_min_m < 0.0:
            raise ValueError("MPC min_speed_forward_min_m must be non-negative")
        if not 0.0 <= self.brake_max <= cfg.BRAKE_MAX:
            raise ValueError("MPC brake_max must be between 0 and BRAKE_MAX")
        if self.reference_horizon_time_s <= 0.0:
            raise ValueError("MPC reference_horizon_time_s must be positive")
        if self.reference_speed_floor_mps < 0.0:
            raise ValueError("MPC reference_speed_floor_mps must be non-negative")
        if self.reference_min_distance_m <= 0.0:
            raise ValueError("MPC reference_min_distance_m must be positive")
        if self.reference_max_distance_m < self.reference_min_distance_m:
            raise ValueError(
                "MPC reference_max_distance_m must be >= reference_min_distance_m"
            )
        if self.reference_smoothing_window < 1:
            raise ValueError("MPC reference_smoothing_window must be positive")
        if self.terminal_slowdown_distance_m < 0.0:
            raise ValueError("MPC terminal_slowdown_distance_m must be non-negative")
        if self.terminal_speed_mps < 0.0:
            raise ValueError("MPC terminal_speed_mps must be non-negative")

    def with_latency_ms(self, latency_ms: float | None) -> "MPCConfig":
        """Return a latency-adjusted config with a longer optimization horizon.

        The base horizon tracks the immediate path. Desktop-local inference delay
        means the control we apply now may be acting on an older trajectory, so we
        add one MPC step per observed latency ``dt`` and bucket the result to the
        configured preset step for stable solve sizes.
        """

        if latency_ms is None:
            return self
        latency_s = max(0.0, float(latency_ms) / 1000.0)
        if not math.isfinite(latency_s) or latency_s <= 0.0:
            return self
        extra_steps = int(math.ceil(latency_s / self.dt))
        target_horizon = self.horizon + extra_steps
        bucket_step = max(1, int(cfg.MPC_LATENCY_PRESET_STEP))
        target_horizon = int(math.ceil(target_horizon / bucket_step) * bucket_step)
        target_horizon = int(
            np.clip(
                max(target_horizon, self.horizon, int(cfg.MPC_LATENCY_PRESET_MIN_HORIZON)),
                int(cfg.MPC_LATENCY_PRESET_MIN_HORIZON),
                int(cfg.MPC_LATENCY_PRESET_MAX_HORIZON),
            )
        )
        return replace(
            self,
            horizon=target_horizon,
            reference_horizon_time_s=float(self.reference_horizon_time_s) + latency_s,
        )


# State: [x, y_left, yaw_left, speed, steer_left_rad, accel_mps2]
IX = 0
IY = 1
IYAW = 2
IV = 3
ISTEER = 4
IACCEL = 5
NX = 6
NU = 2  # [steer_left_cmd_rad, accel_cmd_mps2]


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clean_trajectory_xy(points: np.ndarray) -> np.ndarray:
    """Remove invalid, duplicate, and isolated large-jump waypoints."""

    xy = np.asarray(points, dtype=np.float64)[:, :2]
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if len(xy) == 0:
        raise ValueError("MPC reference trajectory must contain finite points")

    xy = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    positive_lengths = segment_lengths[segment_lengths > 1e-3]
    max_step = np.inf
    if len(positive_lengths) > 0:
        max_step = max(5.0, 10.0 * float(np.median(positive_lengths)))

    cleaned = [xy[0]]
    for point in xy[1:]:
        step = float(np.linalg.norm(point - cleaned[-1]))
        if step <= 1e-3:
            continue
        if step > max_step:
            continue
        cleaned.append(point)

    if len(cleaned) == 1:
        cleaned.append(cleaned[0].copy())
    return np.asarray(cleaned, dtype=np.float64)


def _smooth_xy_by_window(xy: np.ndarray, window: int) -> np.ndarray:
    """Apply a small centered moving-average smooth while preserving endpoints."""

    window = int(window)
    if window < 3 or len(xy) < window:
        return xy
    if window % 2 == 0:
        window += 1
    if len(xy) < window:
        return xy

    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = xy.copy()
    for dim in range(2):
        padded = np.pad(xy[:, dim], (pad, pad), mode="edge")
        smoothed[:, dim] = np.convolve(padded, kernel, mode="valid")
    smoothed[0] = xy[0]
    smoothed[-1] = xy[-1]
    return smoothed


def _resample_xy_by_arclength(
    xy: np.ndarray,
    *,
    num_points: int,
    max_distance_m: float,
) -> np.ndarray:
    """Resample a local path by cumulative arc length."""

    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(arc_lengths[-1])
    if total_length <= 1e-6:
        return np.repeat(xy[:1], num_points, axis=0)

    end_distance = min(float(max_distance_m), total_length)
    target_arc = np.linspace(0.0, end_distance, num_points)
    return np.column_stack(
        [
            np.interp(target_arc, arc_lengths, xy[:, 0]),
            np.interp(target_arc, arc_lengths, xy[:, 1]),
        ]
    )


def resample_reference(
    wp_ego: np.ndarray,
    config: MPCConfig,
    current_speed_mps: float = 0.0,
) -> np.ndarray:
    """Return compact arc-length MPC reference states from Alpamayo trajectory.

    Alpamayo uses an ego frame with positive y to the left. The MPC keeps that
    convention internally, then maps steering sign to CARLA at the output edge.
    The raw 64 Alpamayo points are cleaned, lightly smoothed, and resampled by
    cumulative arc length into ``config.horizon`` MPC steps. Horizon distance is
    speed-dependent and bounded by the configured min/max reference distances.
    Returned state columns are ``[x, y_left, yaw_left, speed]``.
    """

    config.validate()
    points = np.asarray(wp_ego, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected trajectory shape (N, >=2), got {points.shape}")
    if len(points) == 0:
        raise ValueError("MPC reference trajectory must not be empty")

    speed_for_horizon = max(float(current_speed_mps), config.reference_speed_floor_mps)
    horizon_distance_m = float(
        np.clip(
            speed_for_horizon * config.reference_horizon_time_s,
            config.reference_min_distance_m,
            config.reference_max_distance_m,
        )
    )

    xy = _clean_trajectory_xy(points)
    xy = _smooth_xy_by_window(xy, config.reference_smoothing_window)
    xy = _resample_xy_by_arclength(
        xy,
        num_points=config.horizon + 1,
        max_distance_m=horizon_distance_m,
    )
    x = xy[:, 0]
    y = xy[:, 1]

    dx = np.gradient(x, config.dt)
    dy = np.gradient(y, config.dt)
    speed = np.hypot(dx, dy)
    if float(np.max(x)) < float(config.min_speed_forward_min_m):
        speed[:] = 0.0
    yaw = np.arctan2(dy, np.maximum(dx, 1e-6))
    yaw[0] = 0.0
    speed[0] = 0.0
    return np.column_stack([x, y, yaw, speed])


def _apply_minimum_reference_speed(
    ref: np.ndarray,
    config: MPCConfig,
) -> tuple[np.ndarray, bool, float, float]:
    """Floor MPC reference speed when the trajectory contains forward progress."""

    ref_forward_m = float(np.max(ref[:, 0]))
    min_speed_mps = float(config.min_speed_kmh) / 3.6
    if min_speed_mps <= 0.0 or ref_forward_m < float(config.min_speed_forward_min_m):
        return ref, False, ref_forward_m, min_speed_mps

    floored = ref.copy()
    floored[1:, 3] = np.maximum(floored[1:, 3], min_speed_mps)
    if config.terminal_slowdown_distance_m > 0.0:
        segment_lengths = np.linalg.norm(np.diff(floored[:, :2], axis=0), axis=1)
        arc_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        remaining = np.maximum(0.0, float(arc_lengths[-1]) - arc_lengths)
        slowdown_ratio = np.clip(
            remaining / float(config.terminal_slowdown_distance_m),
            0.0,
            1.0,
        )
        terminal_speed = float(config.terminal_speed_mps)
        floored[:, 3] = terminal_speed + (floored[:, 3] - terminal_speed) * slowdown_ratio
    return floored, True, ref_forward_m, min_speed_mps


class MPCFollower:
    """Kinematic linear MPC follower with the same call shape as PID follower."""

    def __init__(self, config: MPCConfig | None = None):
        self.config = config or MPCConfig.from_module_config()
        self.config.validate()
        self._osqp = _import_osqp()
        self._prev_u = np.zeros(NU, dtype=np.float64)
        self._last_status = "not_run"

    def reset(self) -> None:
        self._prev_u[:] = 0.0
        self._last_status = "reset"

    def record_applied_control(self, steer_carla: float, throttle: float, brake: float) -> None:
        """Synchronize MPC input state with the smoothed CARLA control actually applied."""

        steer_carla = float(np.clip(steer_carla, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, cfg.THROTTLE_MAX))
        brake = float(np.clip(brake, 0.0, cfg.BRAKE_MAX))
        steer_left_rad = -steer_carla * float(self.config.max_steer_rad)

        if throttle >= brake:
            accel = throttle / max(float(cfg.THROTTLE_MAX), 1e-6) * self.config.accel_max
        else:
            accel = -brake / max(float(cfg.BRAKE_MAX), 1e-6) * abs(self.config.decel_max)

        self._prev_u[:] = [
            float(np.clip(steer_left_rad, -self.config.max_steer_rad, self.config.max_steer_rad)),
            float(np.clip(accel, self.config.decel_max, self.config.accel_max)),
        ]

    def compute_control(self, _vehicle_tf, wp_ego, speed_mps, latency_ms: float | None = None):
        start = time.perf_counter()
        control_config = self.config.with_latency_ms(latency_ms)
        try:
            control_config.validate()
            ref = resample_reference(wp_ego, control_config, current_speed_mps=speed_mps)
            ref, min_speed_applied, ref_forward_m, min_speed_mps = (
                _apply_minimum_reference_speed(ref, control_config)
            )
            x0 = np.zeros(NX, dtype=np.float64)
            x0[IV] = max(0.0, float(speed_mps))
            x0[ISTEER] = float(self._prev_u[0])
            # The acceleration state is a lagged command. At standstill, carrying
            # a previous negative command makes the first predicted speed
            # negative before the optimizer can affect it, which violates the
            # speed >= 0 constraint and can make an otherwise valid problem
            # infeasible. Clamp the lagged state to what can keep the first
            # predicted speed non-negative.
            x0[IACCEL] = max(float(self._prev_u[1]), -x0[IV] / control_config.dt)
            u_opt, status = self._solve(ref, x0, control_config)
        except Exception as exc:
            self._prev_u[:] = 0.0
            solve_time_ms = (time.perf_counter() - start) * 1000.0
            return 0.0, 0.0, 1.0, {
                "mode": "mpc_error",
                "status": type(exc).__name__,
                "error": str(exc),
                "solve_time_ms": solve_time_ms,
                "latency_ms": None if latency_ms is None else float(latency_ms),
                "base_horizon": int(self.config.horizon),
                "horizon": int(control_config.horizon),
            }

        solve_time_ms = (time.perf_counter() - start) * 1000.0
        self._last_status = status
        if status not in {"solved", "solved_inaccurate"}:
            self._prev_u[:] = 0.0
            if "infeasible" in status:
                return 0.0, 0.0, 0.0, {
                    "mode": "mpc",
                    "status": status,
                    "solve_time_ms": solve_time_ms,
                    "fallback": "coast",
                    "min_speed_applied": min_speed_applied,
                    "min_speed_mps": min_speed_mps,
                    "ref_forward_m": ref_forward_m,
                    "latency_ms": None if latency_ms is None else float(latency_ms),
                    "base_horizon": int(self.config.horizon),
                    "horizon": int(control_config.horizon),
                }
            return 0.0, 0.0, 1.0, {
                "mode": "mpc",
                "status": status,
                "solve_time_ms": solve_time_ms,
                "fallback": "brake",
                "min_speed_applied": min_speed_applied,
                "min_speed_mps": min_speed_mps,
                "ref_forward_m": ref_forward_m,
                "latency_ms": None if latency_ms is None else float(latency_ms),
                "base_horizon": int(self.config.horizon),
                "horizon": int(control_config.horizon),
            }

        steer_left_rad = float(
            np.clip(u_opt[0], -control_config.max_steer_rad, control_config.max_steer_rad)
        )
        accel = float(np.clip(u_opt[1], control_config.decel_max, control_config.accel_max))
        self._prev_u[:] = [steer_left_rad, accel]

        steer_carla = float(
            np.clip(-steer_left_rad / control_config.max_steer_rad, -1.0, 1.0)
        )
        if accel >= 0.0:
            throttle = float(
                np.clip(
                    accel / control_config.accel_max * cfg.THROTTLE_MAX,
                    0.0,
                    cfg.THROTTLE_MAX,
                )
            )
            launch_assist = (
                x0[IV] < cfg.MPC_LAUNCH_SPEED_THRESHOLD_MPS
                and accel > 0.0
                and ref_forward_m >= cfg.MPC_LAUNCH_FORWARD_MIN_M
            )
            if launch_assist:
                throttle = float(
                    np.clip(
                        max(throttle, cfg.MPC_LAUNCH_THROTTLE_MIN),
                        0.0,
                        cfg.THROTTLE_MAX,
                    )
                )
            brake = 0.0
        else:
            launch_assist = False
            throttle = 0.0
            raw_brake = float(
                np.clip(
                    -accel / abs(control_config.decel_max) * cfg.BRAKE_MAX,
                    0.0,
                    cfg.BRAKE_MAX,
                )
            )
            brake = float(np.clip(raw_brake, 0.0, control_config.brake_max))

        return steer_carla, throttle, brake, {
            "mode": "mpc",
            "status": status,
            "solve_time_ms": solve_time_ms,
            "latency_ms": None if latency_ms is None else float(latency_ms),
            "base_horizon": int(self.config.horizon),
            "horizon": int(control_config.horizon),
            "steer_left_rad": steer_left_rad,
            "accel_mps2": accel,
            "launch_assist": launch_assist,
            "min_speed_applied": min_speed_applied,
            "min_speed_mps": min_speed_mps,
            "min_speed_forward_min_m": float(control_config.min_speed_forward_min_m),
            "ref_forward_m": ref_forward_m,
            "ref_final_xy": [float(ref[-1, 0]), float(ref[-1, 1])],
            "raw_brake": raw_brake if accel < 0.0 else 0.0,
            "brake_limited": accel < 0.0 and raw_brake > brake,
        }

    def _linearize_dynamics(self, x_op: np.ndarray, config: MPCConfig) -> tuple[np.ndarray, np.ndarray]:
        c = config
        yaw = float(x_op[IYAW])
        speed = max(0.05, float(x_op[IV]))
        steer = float(np.clip(x_op[ISTEER], -c.max_steer_rad, c.max_steer_rad))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        cos_steer = max(0.2, math.cos(steer))

        a = np.zeros((NX, NX), dtype=np.float64)
        b = np.zeros((NX, NU), dtype=np.float64)
        a[IX, IYAW] = -speed * sin_yaw
        a[IX, IV] = cos_yaw
        a[IY, IYAW] = speed * cos_yaw
        a[IY, IV] = sin_yaw
        a[IYAW, IV] = math.tan(steer) / c.wheelbase_m
        a[IYAW, ISTEER] = speed / (c.wheelbase_m * cos_steer * cos_steer)
        a[IV, IACCEL] = 1.0
        a[ISTEER, ISTEER] = -1.0 / c.steering_tau
        a[IACCEL, IACCEL] = -1.0 / c.accel_tau
        b[ISTEER, 0] = 1.0 / c.steering_tau
        b[IACCEL, 1] = 1.0 / c.accel_tau

        return np.eye(NX, dtype=np.float64) + c.dt * a, c.dt * b

    def _solve_sequence(self, ref: np.ndarray, x0: np.ndarray, config: MPCConfig) -> tuple[np.ndarray, str]:
        c = config
        n = c.horizon
        a_d, b_d = self._linearize_dynamics(x0, c)
        sx, su = _prediction_matrices(a_d, b_d, n)

        x_ref = np.zeros((n + 1, NX), dtype=np.float64)
        x_ref[:, IX] = ref[:, 0]
        x_ref[:, IY] = ref[:, 1]
        x_ref[:, IYAW] = ref[:, 2]
        x_ref[:, IV] = ref[:, 3]
        x_ref[:, IYAW] = _wrap_angle(x_ref[:, IYAW])

        q_state = np.diag([c.w_lon, c.w_lat, c.w_heading, c.w_speed, 0.0, 0.0])
        qf_state = q_state.copy()
        q_blk = sparse.block_diag([q_state] * n + [qf_state], format="csc")
        r_abs = sparse.block_diag([np.diag([c.w_steer, c.w_accel])] * n, format="csc")
        du_matrix = _delta_u_matrix(n)
        rd_blk = sparse.block_diag([np.diag([c.w_dsteer, c.w_daccel])] * n, format="csc")
        du_offset = np.zeros(n * NU, dtype=np.float64)
        du_offset[:NU] = -self._prev_u

        x_free = sx @ x0
        x_ref_flat = x_ref.reshape(-1)
        h = su.T @ q_blk @ su + r_abs + du_matrix.T @ rd_blk @ du_matrix
        g = su.T @ q_blk @ (x_free - x_ref_flat) + du_matrix.T @ rd_blk @ du_offset
        h = sparse.csc_matrix(0.5 * (h + h.T) + 1e-7 * sparse.eye(n * NU, format="csc"))

        a_ineq, lower, upper = self._constraints(su, x_free, c)
        solver = self._osqp.OSQP()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            solver.setup(
                P=h,
                q=np.asarray(g).reshape(-1),
                A=a_ineq,
                l=lower,
                u=upper,
                verbose=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=1000,
                polishing=False,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            result = solver.solve()
        if result.info.status not in {"solved", "solved_inaccurate"} or result.x is None:
            return np.zeros((n, NU), dtype=np.float64), result.info.status
        return np.asarray(result.x, dtype=np.float64).reshape(n, NU), result.info.status

    def _solve(self, ref: np.ndarray, x0: np.ndarray, config: MPCConfig) -> tuple[np.ndarray, str]:
        u_seq, status = self._solve_sequence(ref, x0, config)
        return np.asarray(u_seq[0], dtype=np.float64), status

    def _constraints(self, su: np.ndarray, x_free: np.ndarray, config: MPCConfig):
        c = config
        n = c.horizon
        a_input = sparse.eye(n * NU, format="csc")
        l_input = np.tile([-c.max_steer_rad, c.decel_max], n)
        u_input = np.tile([c.max_steer_rad, c.accel_max], n)

        constrained_rows = []
        lower_state = []
        upper_state = []
        for k in range(1, n + 1):
            base = k * NX
            constrained_rows.extend([base + IV, base + ISTEER, base + IACCEL])
            lower_state.extend([0.0, -c.max_steer_rad, c.decel_max])
            upper_state.extend([50.0, c.max_steer_rad, c.accel_max])

        su_state = sparse.csc_matrix(su[constrained_rows, :])
        x_free_state = x_free[constrained_rows]
        lower_state = np.asarray(lower_state, dtype=np.float64) - x_free_state
        upper_state = np.asarray(upper_state, dtype=np.float64) - x_free_state
        a_ineq = sparse.vstack([a_input, su_state], format="csc")
        lower = np.concatenate([l_input, lower_state])
        upper = np.concatenate([u_input, upper_state])
        return a_ineq, lower, upper


def _prediction_matrices(
    a_d: np.ndarray,
    b_d: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    sx = np.zeros(((horizon + 1) * NX, NX), dtype=np.float64)
    su = np.zeros(((horizon + 1) * NX, horizon * NU), dtype=np.float64)
    a_powers = [np.eye(NX, dtype=np.float64)]
    for _ in range(horizon):
        a_powers.append(a_d @ a_powers[-1])
    for k in range(horizon + 1):
        sx[k * NX : (k + 1) * NX] = a_powers[k]
        for j in range(k):
            su[k * NX : (k + 1) * NX, j * NU : (j + 1) * NU] = (
                a_powers[k - 1 - j] @ b_d
            )
    return sx, su


def _delta_u_matrix(horizon: int):
    mat = sparse.lil_matrix((horizon * NU, horizon * NU), dtype=np.float64)
    for k in range(horizon):
        row = k * NU
        col = k * NU
        mat[row : row + NU, col : col + NU] = np.eye(NU)
        if k > 0:
            mat[row : row + NU, col - NU : col] = -np.eye(NU)
    return mat.tocsc()
