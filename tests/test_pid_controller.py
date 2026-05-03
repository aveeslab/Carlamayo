import numpy as np

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
