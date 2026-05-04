from module.respawn_control import RespawnMonitor


def test_respawn_monitor_triggers_on_new_collision_once():
    monitor = RespawnMonitor(cooldown_frames=10)

    decision = monitor.check_collision(
        frame_count=12,
        collision_count=1,
        last_collision_event={"other_actor": "vehicle.audi.tt", "intensity": 42.0},
    )

    assert decision.should_respawn is True
    assert "collision" in decision.reason
    assert "vehicle.audi.tt" in decision.reason

    monitor.mark_respawn(frame_count=12, collision_count=1)
    repeat = monitor.check_collision(
        frame_count=13,
        collision_count=1,
        last_collision_event={"other_actor": "vehicle.audi.tt", "intensity": 42.0},
    )

    assert repeat.should_respawn is False


def test_respawn_monitor_respects_collision_cooldown():
    monitor = RespawnMonitor(cooldown_frames=10)
    monitor.mark_respawn(frame_count=20, collision_count=1)

    decision = monitor.check_collision(
        frame_count=25,
        collision_count=2,
        last_collision_event={"other_actor": "vehicle.audi.tt", "intensity": 42.0},
    )

    assert decision.should_respawn is False


def test_respawn_monitor_allows_collision_after_cooldown():
    monitor = RespawnMonitor(cooldown_frames=10)
    monitor.mark_respawn(frame_count=20, collision_count=1)

    decision = monitor.check_collision(
        frame_count=30,
        collision_count=2,
        last_collision_event={"other_actor": "vehicle.audi.tt", "intensity": 42.0},
    )

    assert decision.should_respawn is True
