import numpy as np

from module import config as cfg
from module.pid_controller import OfficialPIDFollower


class RaisingMap:
    def get_waypoint(self, *args, **kwargs):
        raise AssertionError("raw Alpamayo control must not project to CARLA map waypoints")


def test_pick_target_uses_raw_alpamayo_path_without_map_projection():
    follower = object.__new__(OfficialPIDFollower)
    follower.map = RaisingMap()
    wp_world = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, -2.0, 0.0],
            [6.0, -4.0, 0.0],
        ],
        dtype=np.float64,
    )

    target_wp, target_idx, _lookahead_m = follower._pick_target(wp_world, speed_mps=0.0)

    assert target_idx == 2
    assert target_wp.transform.location.x == wp_world[target_idx, 0]
    assert target_wp.transform.location.y == wp_world[target_idx, 1]
    assert target_wp.transform.location.z == wp_world[target_idx, 2]


def test_target_speed_estimate_allows_stop_trajectory():
    wp_local = np.zeros((12, 3), dtype=np.float64)

    target_speed = OfficialPIDFollower._estimate_target_speed_kmh(wp_local)

    assert target_speed == 0.0


def test_target_speed_estimate_allows_tiny_stop_trajectory():
    wp_local = np.zeros((12, 3), dtype=np.float64)
    wp_local[:, 0] = np.linspace(0.0, cfg.PID_STOP_TRAJECTORY_MAX_EXTENT_M * 0.5, 12)

    target_speed = OfficialPIDFollower._estimate_target_speed_kmh(wp_local)

    assert target_speed == 0.0


def test_target_speed_estimate_uses_trajectory_displacement_rate():
    wp_local = np.zeros((12, 3), dtype=np.float64)
    wp_local[:, 0] = np.arange(12) * 0.5

    target_speed = OfficialPIDFollower._estimate_target_speed_kmh(wp_local)

    assert 17.5 <= target_speed <= 18.5


def test_target_speed_estimate_keeps_dense_forward_path_moving():
    wp_local = np.zeros((64, 3), dtype=np.float64)
    wp_local[:10, 0] = np.linspace(0.0, 0.02, 10)
    wp_local[10:, 0] = np.linspace(0.02, 6.0, 54)

    target_speed = OfficialPIDFollower._estimate_target_speed_kmh(wp_local)

    assert target_speed >= cfg.PID_TARGET_SPEED_MIN_KMH


def test_steering_gain_reduces_pid_output():
    scaled = OfficialPIDFollower._scale_steering(0.8)

    assert scaled == cfg.STEERING_GAIN * 0.8
    assert abs(scaled) < 0.8


def test_steering_gain_preserves_sign_and_zero():
    assert OfficialPIDFollower._scale_steering(-0.5) < 0.0
    assert OfficialPIDFollower._scale_steering(0.0) == 0.0
