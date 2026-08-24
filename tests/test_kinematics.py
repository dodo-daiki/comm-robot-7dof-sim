import numpy as np
import pytest

from robot.joints import load_default
from robot.kinematics import forward_kinematics, world_position


@pytest.fixture
def config():
    return load_default()


def test_zero_pose_end_effector_positions(config):
    transforms = forward_kinematics(config, {})

    base = np.array([0, 0, config.base_height_mm], dtype=float)

    waist = config.joints["waist_yaw"]
    neck_yaw = config.joints["neck_yaw"]
    neck_pitch = config.joints["neck_pitch"]
    shoulder_l = config.joints["shoulder_pitch_l"]
    elbow_l = config.joints["elbow_pitch_l"]
    shoulder_r = config.joints["shoulder_pitch_r"]
    elbow_r = config.joints["elbow_pitch_r"]

    expected_head = (
        base
        + waist.origin_offset_mm
        + neck_yaw.origin_offset_mm
        + neck_pitch.origin_offset_mm
        + neck_pitch.end_effector_offset_mm
    )
    expected_hand_l = (
        base
        + waist.origin_offset_mm
        + shoulder_l.origin_offset_mm
        + elbow_l.origin_offset_mm
        + elbow_l.end_effector_offset_mm
    )
    expected_hand_r = (
        base
        + waist.origin_offset_mm
        + shoulder_r.origin_offset_mm
        + elbow_r.origin_offset_mm
        + elbow_r.end_effector_offset_mm
    )

    np.testing.assert_allclose(world_position(transforms["head"]), expected_head)
    np.testing.assert_allclose(world_position(transforms["hand_l"]), expected_hand_l)
    np.testing.assert_allclose(world_position(transforms["hand_r"]), expected_hand_r)


def test_waist_yaw_rotates_upper_body(config):
    zero_transforms = forward_kinematics(config, {})
    hand_l_zero = world_position(zero_transforms["hand_l"])
    hand_r_zero = world_position(zero_transforms["hand_r"])

    rotated_transforms = forward_kinematics(config, {"waist_yaw": 90.0})
    hand_l_rotated = world_position(rotated_transforms["hand_l"])
    hand_r_rotated = world_position(rotated_transforms["hand_r"])

    base_z = config.base_height_mm
    theta = np.radians(90.0)
    c, s = np.cos(theta), np.sin(theta)

    def rotate_about_waist_z(point):
        x, y, z = point
        x -= 0.0
        y -= 0.0
        rel_z = z - base_z
        rx = x * c - y * s
        ry = x * s + y * c
        return np.array([rx, ry, base_z + rel_z])

    expected_hand_l = rotate_about_waist_z(hand_l_zero)
    expected_hand_r = rotate_about_waist_z(hand_r_zero)

    np.testing.assert_allclose(hand_l_rotated, expected_hand_l, atol=1e-9)
    np.testing.assert_allclose(hand_r_rotated, expected_hand_r, atol=1e-9)

    assert hand_l_rotated[0] == pytest.approx(0.0, abs=1e-9)
    assert hand_l_rotated[1] < 0
    assert hand_r_rotated[0] == pytest.approx(0.0, abs=1e-9)
    assert hand_r_rotated[1] > 0


def test_out_of_range_angle_is_clamped_not_raised(config):
    lo, hi = config.joints["elbow_pitch_l"].range_deg

    with pytest.warns(UserWarning):
        transforms = forward_kinematics(config, {"elbow_pitch_l": hi + 1000.0})

    clamped_transforms = forward_kinematics(config, {"elbow_pitch_l": hi})

    np.testing.assert_allclose(
        world_position(transforms["hand_l"]),
        world_position(clamped_transforms["hand_l"]),
        atol=1e-9,
    )

    with pytest.warns(UserWarning):
        low_transforms = forward_kinematics(config, {"elbow_pitch_l": lo - 1000.0})

    clamped_low_transforms = forward_kinematics(config, {"elbow_pitch_l": lo})

    np.testing.assert_allclose(
        world_position(low_transforms["hand_l"]),
        world_position(clamped_low_transforms["hand_l"]),
        atol=1e-9,
    )
