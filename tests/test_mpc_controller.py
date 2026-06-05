import numpy as np

from module import config as cfg
from module.mpc_controller import MPCConfig, MPCFollower, resample_reference


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
