from module.respawn_control import RespawnMonitor


def test_collision_monitor_ignores_already_seen_collision_count():
    monitor = RespawnMonitor(cooldown_frames=10)
    monitor.mark_respawn(frame_count=20, collision_count=2)

    decision = monitor.check_collision(
        frame_count=30,
        collision_count=2,
        last_collision_event={"other_actor": "vehicle", "intensity": 8.0},
    )

    assert not decision.should_respawn
    assert decision.reason == ""


def test_collision_monitor_reports_new_collision_with_details_after_cooldown():
    monitor = RespawnMonitor(cooldown_frames=10)
    monitor.mark_respawn(frame_count=5, collision_count=0)

    decision = monitor.check_collision(
        frame_count=15,
        collision_count=1,
        last_collision_event={"other_actor": "vehicle.tesla.model3", "intensity": 12.34},
    )

    assert decision.should_respawn
    assert decision.reason == "collision detected (actor=vehicle.tesla.model3, impulse=12.3)"


def test_collision_monitor_suppresses_respawn_during_cooldown_but_marks_collision_seen():
    monitor = RespawnMonitor(cooldown_frames=10)
    monitor.mark_respawn(frame_count=100, collision_count=0)

    first = monitor.check_collision(frame_count=105, collision_count=1, last_collision_event=None)
    second = monitor.check_collision(frame_count=115, collision_count=1, last_collision_event=None)

    assert not first.should_respawn
    assert not second.should_respawn


def test_collision_monitor_respawns_first_collision_without_prior_respawn():
    monitor = RespawnMonitor(cooldown_frames=10)

    decision = monitor.check_collision(
        frame_count=1,
        collision_count=1,
        last_collision_event={"other_actor_type": "walker.pedestrian", "intensity": 1},
    )

    assert decision.should_respawn
    assert decision.reason == "collision detected (actor=walker.pedestrian, impulse=1.0)"
