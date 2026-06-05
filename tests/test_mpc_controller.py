from dataclasses import replace

import numpy as np

from module import config as cfg
from module.mpc_controller import (
    MPCConfig,
    MPCFollower,
    _apply_minimum_reference_speed,
    resample_reference,
)


def test_mpc_config_loads_tuning_values_from_module_config():
    config = MPCConfig.from_module_config()

    assert config.dt == cfg.MPC_DT
    assert config.horizon == cfg.MPC_HORIZON
    assert config.max_steer_rad == cfg.MAX_STEER_RAD
    assert config.reference_max_distance_m == cfg.MPC_REFERENCE_MAX_DISTANCE_M


def test_mpc_reference_resampling_preserves_alpamayo_left_positive_coordinates():
    config = MPCConfig.from_module_config()
    wp_ego = np.array(
        [
            [2.0, 0.5, 0.0],
            [4.0, 1.0, 0.0],
            [6.0, 1.5, 0.0],
        ],
        dtype=np.float64,
    )

    reference = resample_reference(wp_ego, config, current_speed_mps=0.0)

    assert reference.shape == (config.horizon + 1, 4)
    assert reference[-1, 0] > 0.0
    assert reference[-1, 1] > 0.0


def test_mpc_follower_solves_simple_forward_reference():
    follower = MPCFollower()
    wp_ego = np.column_stack(
        [
            np.linspace(2.0, 30.0, 16),
            np.zeros(16),
            np.zeros(16),
        ]
    )

    steer, throttle, brake, debug = follower.compute_control(
        None,
        wp_ego,
        speed_mps=0.0,
    )

    assert debug["mode"] == "mpc"
    assert debug["status"] in {"solved", "solved_inaccurate"}
    assert -1.0 <= steer <= 1.0
    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0


def test_mpc_records_smoothed_applied_control_as_previous_input():
    follower = MPCFollower()

    follower.record_applied_control(steer_carla=0.25, throttle=0.1, brake=0.0)

    assert follower._prev_u[0] == -0.25 * cfg.MAX_STEER_RAD
    assert follower._prev_u[1] == 0.1 / cfg.THROTTLE_MAX * cfg.ACCEL_MAX

    follower.record_applied_control(steer_carla=0.0, throttle=0.0, brake=cfg.BRAKE_MAX)

    assert follower._prev_u[0] == 0.0
    assert follower._prev_u[1] == cfg.DECEL_MAX


def test_mpc_reference_speed_tapers_near_terminal_path_point():
    config = replace(
        MPCConfig.from_module_config(),
        min_speed_kmh=18.0,
        terminal_slowdown_distance_m=4.0,
        terminal_speed_mps=0.0,
    )
    ref = np.column_stack(
        [
            np.linspace(0.0, 10.0, 6),
            np.zeros(6),
            np.zeros(6),
            np.ones(6),
        ]
    )

    tapered, min_speed_applied, _ref_forward_m, min_speed_mps = (
        _apply_minimum_reference_speed(ref, config)
    )

    assert min_speed_applied is True
    assert tapered[1, 3] == min_speed_mps
    assert 0.0 < tapered[-2, 3] < min_speed_mps
    assert tapered[-1, 3] == 0.0
