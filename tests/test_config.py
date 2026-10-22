from pathlib import Path

from module import config as cfg


def test_runtime_dimensions_match_alpamayo_camera_history_contract():
    # The camera count is version-specific and now lives on the adapter, not here.
    assert cfg.NUM_FRAMES == 4
    assert cfg.NUM_HISTORY == 16
    assert cfg.IMG_CHANNELS == 3
    assert cfg.IMG_WIDTH > 0
    assert cfg.IMG_HEIGHT > 0


def test_per_loop_output_video_names_are_bare_mp4_filenames():
    from pathlib import Path as _Path

    for name in (cfg.OUTPUT_VIDEO, cfg.OPEN_LOOP_OUTPUT_VIDEO, cfg.LIVE_OPEN_LOOP_OUTPUT_VIDEO):
        assert name.endswith(".mp4")
        assert _Path(name).name == name


def test_output_paths_and_map_defaults_are_public_run_defaults():
    assert cfg.CARLA_MAP == "Town03"
    assert cfg.OUTPUT_VIDEO.endswith(".mp4")
    assert Path(cfg.OUTPUT_VIDEO).name == cfg.OUTPUT_VIDEO
    assert cfg.VIDEO_FPS > 0
    assert cfg.PYGAME_WINDOW_WIDTH > 0
    assert cfg.PYGAME_WINDOW_HEIGHT > 0


def test_control_and_respawn_limits_are_safe_positive_ranges():
    assert 0.0 < cfg.CONTROL_DT <= 1.0
    assert 0.0 <= cfg.THROTTLE_MAX <= 1.0
    assert 0.0 <= cfg.BRAKE_MAX <= 1.0
    assert cfg.FOLLOW_TARGET_MEDIAN_TICKS >= 1
    assert cfg.RESPAWN_COLLISION_COOLDOWN_FRAMES > 0


def test_npc_exclusion_keywords_are_normalized_strings():
    assert cfg.NPC_EXCLUDED_VEHICLE_KEYWORDS
    assert all(keyword == keyword.lower() for keyword in cfg.NPC_EXCLUDED_VEHICLE_KEYWORDS)
    assert all(keyword.strip() == keyword for keyword in cfg.NPC_EXCLUDED_VEHICLE_KEYWORDS)
