import numpy as np
import pytest

from robot.joints import load_default
from robot.kinematics import forward_kinematics, world_position


@pytest.fixture
def config():
    return load_default()


def test_zero_pose_end_effector_positions(config):
    transforms = forward_kinematics(config, {})

    np.testing.assert_allclose(world_position(transforms["head"]), [0.0, 0.0, 960.0])
    np.testing.assert_allclose(world_position(transforms["hand_l"]), [-150.0, 0.0, 320.0])
    np.testing.assert_allclose(world_position(transforms["hand_r"]), [150.0, 0.0, 320.0])


def test_waist_yaw_rotates_upper_body(config):
    rotated_transforms = forward_kinematics(config, {"waist_yaw": 90.0})
    hand_l_rotated = world_position(rotated_transforms["hand_l"])
    hand_r_rotated = world_position(rotated_transforms["hand_r"])

    # Zero-pose hand_l is at (-150, 0, 320); zero-pose hand_r is at (150, 0, 320).
    # A +90 deg rotation about waist Z maps (x, y) -> (x*cos90 - y*sin90, x*sin90 + y*cos90).
    np.testing.assert_allclose(hand_l_rotated, [0.0, -150.0, 320.0], atol=1e-9)
    np.testing.assert_allclose(hand_r_rotated, [0.0, 150.0, 320.0], atol=1e-9)


def test_shoulder_pitch_moves_hand_forward_not_lateral(config):
    # In a Z-up/X-right/Y-forward frame, pitch (rotation about X) must move the
    # hand in the sagittal Y-Z plane and must NEVER change X, since X is the
    # only coordinate separating the left/right sides in this pitch-only
    # (no-roll) shoulder design. This directly guards against issue 1
    # (wrong pitch axis) and proves issue 2 (arm crossing the torso) cannot
    # happen once the axis is correct: same-sign shoulder commands keep each
    # hand's X fixed at its own side, so left/right motion is automatically
    # mirror-symmetric without needing an opposite-sign convention.
    transforms = forward_kinematics(config, {"shoulder_pitch_l": 90.0, "shoulder_pitch_r": 90.0})
    hand_l = world_position(transforms["hand_l"])
    hand_r = world_position(transforms["hand_r"])

    np.testing.assert_allclose(hand_l, [-150.0, 480.0, 800.0], atol=1e-9)
    np.testing.assert_allclose(hand_r, [150.0, 480.0, 800.0], atol=1e-9)

    # Mirror symmetry: same X magnitude (opposite sign), identical Y and Z.
    assert hand_l[0] == pytest.approx(-hand_r[0])
    assert hand_l[1] == pytest.approx(hand_r[1])
    assert hand_l[2] == pytest.approx(hand_r[2])
    # Moved forward (+Y) from the zero-pose Y of 0, not sideways.
    assert hand_l[1] > 0
    assert hand_r[1] > 0


def test_rotated_parent_with_offset_child_matches_hand_computed_pose(config):
    # Regression test for composition-order bugs (e.g. reversing parent/child
    # multiplication order, or swapping translate/rotate order inside
    # joint_transform). waist_yaw=30 gives shoulder_pitch_l a rotated parent
    # frame, and shoulder_pitch_l=40 / elbow_pitch_l=25 both rotate a joint
    # that has a nonzero origin_offset_mm feeding a further offset downstream
    # -- none of these offsets are aligned with a rotation axis, so any
    # composition-order bug shows up as a large positional error.
    #
    # Expected value derived independently (NOT via robot.kinematics) using
    # a from-scratch script implementing the same rotation matrices by hand:
    #   R_waist = Rz(30); p_waist = (0, 0, 500)
    #   p_shoulder = p_waist + R_waist @ (-150, 0, 300)
    #   R_shoulder = R_waist @ Rx(40)
    #   p_elbow = p_shoulder + R_shoulder @ (0, 0, -250)
    #   R_elbow = R_shoulder @ Rx(25)
    #   p_hand = p_elbow + R_elbow @ (0, 0, -230)
    transforms = forward_kinematics(
        config,
        {"waist_yaw": 30.0, "shoulder_pitch_l": 40.0, "elbow_pitch_l": 25.0},
    )
    hand_l = world_position(transforms["hand_l"])

    expected_hand_l = [-314.477657287698, 244.691280267526, 511.286689019895]
    np.testing.assert_allclose(hand_l, expected_hand_l, atol=1e-6)


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


def test_zero_length_axis_raises():
    from robot.kinematics import joint_transform

    with pytest.raises(ValueError):
        joint_transform(np.array([0.0, 0.0, 0.0]), 10.0, np.array([0.0, 0.0, 0.0]))


def test_duplicate_root_raises():
    from robot.joints import RobotConfig

    data = {
        "coordinate_frame": "test",
        "base_height_mm": 0,
        "joints": [
            {
                "name": "a",
                "parent": None,
                "part_ja": "a",
                "axis": [0, 0, 1],
                "origin_offset_mm": [0, 0, 0],
                "range_deg": [-1, 1],
                "motor": "m",
                "gear_ratio": 1,
                "can_id": "TODO",
            },
            {
                "name": "b",
                "parent": None,
                "part_ja": "b",
                "axis": [0, 0, 1],
                "origin_offset_mm": [0, 0, 0],
                "range_deg": [-1, 1],
                "motor": "m",
                "gear_ratio": 1,
                "can_id": "TODO",
            },
        ],
    }
    with pytest.raises(ValueError):
        RobotConfig(data)


def test_unknown_parent_raises():
    from robot.joints import RobotConfig

    data = {
        "coordinate_frame": "test",
        "base_height_mm": 0,
        "joints": [
            {
                "name": "a",
                "parent": None,
                "part_ja": "a",
                "axis": [0, 0, 1],
                "origin_offset_mm": [0, 0, 0],
                "range_deg": [-1, 1],
                "motor": "m",
                "gear_ratio": 1,
                "can_id": "TODO",
            },
            {
                "name": "b",
                "parent": "does_not_exist",
                "part_ja": "b",
                "axis": [0, 0, 1],
                "origin_offset_mm": [0, 0, 0],
                "range_deg": [-1, 1],
                "motor": "m",
                "gear_ratio": 1,
                "can_id": "TODO",
            },
        ],
    }
    with pytest.raises(ValueError):
        RobotConfig(data)


def test_end_effector_name_without_offset_raises():
    from robot.joints import RobotConfig

    data = {
        "coordinate_frame": "test",
        "base_height_mm": 0,
        "joints": [
            {
                "name": "a",
                "parent": None,
                "part_ja": "a",
                "axis": [0, 0, 1],
                "origin_offset_mm": [0, 0, 0],
                "range_deg": [-1, 1],
                "motor": "m",
                "gear_ratio": 1,
                "can_id": "TODO",
                "end_effector_name": "tip",
            },
        ],
    }
    with pytest.raises(ValueError):
        RobotConfig(data)
