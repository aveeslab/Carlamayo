import numpy as np

from module.visualization import project_trajectory_to_image


def test_project_trajectory_to_image_draws_control_target_overlay():
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    pred_xyz = np.array([[8.0, 0.0, 0.0], [16.0, 0.0, 0.0]], dtype=np.float32)
    control_target_xyz = np.array([12.0, -2.0, 0.0], dtype=np.float32)

    result = project_trajectory_to_image(
        image,
        pred_xyz,
        selected_idx=0,
        control_target_xyz=control_target_xyz,
    )

    green_dominant = (
        (result[:, :, 1] > 0)
        & (result[:, :, 1] > result[:, :, 0])
        & (result[:, :, 1] > result[:, :, 2])
    )
    assert green_dominant.any()
