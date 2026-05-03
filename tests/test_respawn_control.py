from module.respawn_control import RespawnMonitor


def test_respawn_monitor_triggers_on_new_collision_once():
    monitor = RespawnMonitor(cooldown_frames=10, stuck_frames=0)

    decision = monitor.check_collision(
        frame_count=12,
        collision_count=1,
        last_collision_event={"other_actor": "vehicle"},
    )

    assert decision.should_respawn is True
    assert "collision" in decision.reason

    monitor.mark_respawn(frame_count=12, collision_count=1)
    repeat = monitor.check_collision(
        frame_count=13,
        collision_count=1,
        last_collision_event={"other_actor": "vehicle"},
    )

    assert repeat.should_respawn is False


def test_respawn_monitor_respects_collision_cooldown():
    monitor = RespawnMonitor(cooldown_frames=10, stuck_frames=0)
    monitor.mark_respawn(frame_count=20, collision_count=1)

    decision = monitor.check_collision(
        frame_count=25,
        collision_count=2,
        last_collision_event={"other_actor": "vehicle"},
    )

    assert decision.should_respawn is False


def test_respawn_monitor_triggers_after_consecutive_stuck_frames():
    monitor = RespawnMonitor(cooldown_frames=0, stuck_frames=3, stuck_speed_kmh=0.5)

    decisions = [
        monitor.check_stuck(
            frame_count=frame,
            speed_kmh=0.1,
            throttle=0.35,
            brake=0.0,
            has_trajectory=True,
        )
        for frame in (10, 11, 12)
    ]

    assert decisions[0].should_respawn is False
    assert decisions[1].should_respawn is False
    assert decisions[2].should_respawn is True
    assert "stuck" in decisions[2].reason


def test_respawn_monitor_clears_stuck_window_when_vehicle_moves():
    monitor = RespawnMonitor(cooldown_frames=0, stuck_frames=3, stuck_speed_kmh=0.5)

    monitor.check_stuck(
        frame_count=10,
        speed_kmh=0.1,
        throttle=0.35,
        brake=0.0,
        has_trajectory=True,
    )
    monitor.check_stuck(
        frame_count=11,
        speed_kmh=1.0,
        throttle=0.35,
        brake=0.0,
        has_trajectory=True,
    )
    decision = monitor.check_stuck(
        frame_count=12,
        speed_kmh=0.1,
        throttle=0.35,
        brake=0.0,
        has_trajectory=True,
    )

    assert decision.should_respawn is False
