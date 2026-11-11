import numpy as np
import pytest

from module.adapters import SUPPORTED_VERSIONS, get_adapter


def test_supported_versions_dispatch_to_expected_models():
    expected = {
        "1": ("nvidia/Alpamayo-R1-10B", 4),
        "1.5": ("nvidia/Alpamayo-1.5-10B", 4),
        "2": ("nvidia/Alpamayo2-Super", 7),
        "2-edge": ("nvidia/Alpamayo2-Edge", 6),
    }
    assert set(SUPPORTED_VERSIONS) == set(expected)
    for version, (model_id, num_cameras) in expected.items():
        adapter = get_adapter(version)
        assert adapter.version == version
        assert adapter.model_id == model_id
        assert adapter.num_cameras == num_cameras


@pytest.mark.parametrize(
    ("alias", "resolved"),
    [
        ("r1", "1"), ("1.0", "1"), ("15", "1.5"), ("super", "2"), ("Alpamayo2", "2"), ("2.0", "2"),
        ("edge", "2-edge"), ("Alpamayo2-Edge", "2-edge"),
    ],
)
def test_version_aliases_resolve(alias, resolved):
    assert get_adapter(alias).version == resolved


def test_unknown_version_raises():
    with pytest.raises(ValueError, match="Unsupported Alpamayo version"):
        get_adapter("3")


def test_capability_matrix():
    r1, a15, a2 = get_adapter("1"), get_adapter("1.5"), get_adapter("2")
    assert (r1.supports_navigation, r1.supports_vqa, r1.supports_oom_free) == (False, False, False)
    assert (a15.supports_navigation, a15.supports_vqa, a15.supports_oom_free) == (True, True, True)
    assert (a2.supports_navigation, a2.supports_vqa, a2.supports_oom_free) == (True, True, False)
    edge = get_adapter("2-edge")
    assert (edge.supports_navigation, edge.supports_vqa, edge.supports_oom_free) == (
        True, True, False
    )


def test_frame_and_history_profiles():
    # 1 / 1.5 / 2 keep four consecutive ticks and 16 history poses.
    for version in ("1", "1.5", "2"):
        adapter = get_adapter(version)
        assert adapter.frame_ticks == (3, 2, 1, 0) and adapter.num_history_steps == 16
    edge = get_adapter("2-edge")
    assert edge.frame_ticks == (20, 10, 5, 3, 2, 1, 0)
    assert edge.num_history_steps == 20
    assert "camera_rear_tele_30fov" not in edge.source_camera_configs
    assert edge.source_camera_indices == (0, 1, 2, 3, 5, 6)


def _edge_input(t0_us=5_000_000):
    edge = get_adapter("2-edge")
    num_ticks = len(edge.frame_ticks)
    # Pixel value encodes (camera slot, tick position) so selections can be checked.
    images = np.zeros((6, num_ticks, 4, 6, 3), dtype=np.uint8)
    for slot in range(6):
        for t in range(num_ticks):
            images[slot, t] = slot * 10 + t
    data = edge.prepare_model_input(
        images, np.zeros((20, 3), np.float32), np.tile(np.eye(3, dtype=np.float32), (20, 1, 1)),
        t0_us,
    )
    return edge, data


def test_edge_driving_sample_follows_the_released_input_profile():
    edge, data = _edge_input()
    sample = edge.driving_sample(data, navigation_text="")

    assert sample["image_frames"].shape == (22, 1, 3, 4, 6)
    assert sample["image_frames"].is_contiguous()
    cameras = sample["camera_indices"].tolist()
    assert {c: cameras.count(c) for c in set(cameras)} == {0: 3, 1: 5, 2: 3, 3: 3, 5: 3, 6: 5}
    stamps = sample["absolute_timestamps"].tolist()
    front = [s for c, s in zip(cameras, stamps) if c == 1]
    side = [s for c, s in zip(cameras, stamps) if c == 3]
    assert front == [3_000_000, 4_000_000, 4_500_000, 4_900_000, 5_000_000]
    assert side == [4_000_000, 4_900_000, 5_000_000]
    # The rear-left camera (rig slot 3) at -1 s is frame_ticks position of tick 10.
    assert sample["image_frames"][cameras.index(3), 0, 0, 0, 0] == 3 * 10 + 1
    assert sample["ego_history_xyz"].shape == (1, 20, 3)
    # Edge requires a navigation instruction; empty input uses the default.
    assert sample["nav_text"] == [edge.default_navigation_text]
    assert edge.driving_sample(data, " Turn left ")["nav_text"] == ["Turn left"]


def test_edge_vqa_inputs_use_four_recent_frames_per_camera():
    edge, data = _edge_input()
    vqa = edge._camera_inputs(data, {c: (3, 2, 1, 0) for c in edge.source_camera_indices})
    assert vqa["image_frames"].shape == (24, 1, 3, 4, 6)
    assert vqa["absolute_timestamps"].tolist()[:4] == [4_700_000, 4_800_000, 4_900_000, 5_000_000]


def test_four_camera_versions_share_front_ring_and_indices():
    r1, a15 = get_adapter("1"), get_adapter("1.5")
    assert list(r1.source_camera_configs) == list(a15.source_camera_configs)
    assert r1.source_camera_indices == (0, 1, 2, 6)
    assert list(r1.source_camera_configs)[r1.viz_camera_slot] == "camera_front_wide_120fov"


def test_seven_camera_ring_matches_canonical_indices():
    a2 = get_adapter("2")
    assert a2.source_camera_indices == (0, 1, 2, 3, 4, 5, 6)
    assert list(a2.source_camera_configs)[a2.viz_camera_slot] == "camera_front_wide_120fov"


def test_r1_rejects_unsupported_features():
    r1 = get_adapter("1")
    with pytest.raises(NotImplementedError):
        r1.run_vqa(model=None, processor=None, data={}, question="q")
    with pytest.raises(NotImplementedError):
        r1.extract_answer_text({"answer": "x"})


def test_cot_extraction_handles_nested_shapes():
    a2 = get_adapter("2")
    assert a2.extract_cot_text({"cot": [[[["  keep lane  "]]]]}) == "keep lane"
    assert a2.extract_cot_text({"cot": np.array(["slow down"], dtype=object)}) == "slow down"
    assert a2.extract_cot_text({}) == ""


def test_answer_extraction_strips_special_tokens():
    a15 = get_adapter("1.5")
    extra = {"answer": np.array(
        ["<|answer_start|>The light is red.<|answer_end|><|im_end|>"], dtype=object
    )}
    assert a15.extract_answer_text(extra) == "The light is red."


def test_rig_fov_keeps_the_calibrated_center_focal_length():
    from module.adapters._rigs import SEVEN_CAMERA_RIG

    # Pinhole focal length (px at 1920 wide) must equal the f-theta calibration.
    fov = SEVEN_CAMERA_RIG["camera_front_wide_120fov"]["fov"]
    assert 960 / np.tan(np.radians(fov / 2)) == pytest.approx(925.540742)
    assert SEVEN_CAMERA_RIG["camera_front_tele_30fov"]["fov"] == pytest.approx(29.2, abs=0.1)
