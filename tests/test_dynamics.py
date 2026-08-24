import numpy as np
import pytest

from robot.dynamics import compute_joint_torques, torque_margin
from robot.joints import load_default
from robot.mass_estimator import estimate_all_link_masses
from robot.motor_specs import get_motor_spec


@pytest.fixture
def config():
    return load_default()


def test_yaw_joints_always_get_zero_gravity_torque(config):
    # General hand-provable fact (not specific to any one pose): gravity acts
    # purely along -Z, so F = (0, 0, -mg) always has Fx = Fy = 0. The Z
    # component of any r x F is (r_x*F_y - r_y*F_x), which is therefore
    # always exactly 0, regardless of r. So the torque about ANY joint whose
    # world-frame rotation axis is exactly (0, 0, 1) must be exactly 0.
    #
    # In this robot, waist_yaw's axis is Z with no parent (base frame has no
    # rotation), and neck_yaw's axis is Z with its only ancestor (waist_yaw)
    # also rotating about Z -- and a Z-rotation leaves the vector (0,0,1)
    # unchanged. So both waist_yaw and neck_yaw must read exactly 0 torque
    # at ANY pose and ANY (nonzero) mass distribution.
    masses = estimate_all_link_masses(config)
    pose = {
        "waist_yaw": 45.0,
        "neck_yaw": 20.0,
        "neck_pitch": 10.0,
        "shoulder_pitch_l": 60.0,
        "elbow_pitch_l": 30.0,
        "shoulder_pitch_r": -60.0,
        "elbow_pitch_r": 30.0,
    }

    torques = compute_joint_torques(config, pose, masses)

    assert torques["waist_yaw"] == pytest.approx(0.0, abs=1e-9)
    assert torques["neck_yaw"] == pytest.approx(0.0, abs=1e-9)


def test_shoulder_torque_matches_hand_computation_with_arm_extended(config):
    # Pose: shoulder_pitch_l = 90 deg, everything else 0. At zero pose the
    # left arm hangs straight down (link parallel to gravity -> zero
    # moment arm); rotating the shoulder 90 deg about its X axis swings the
    # forearm out to +Y, giving a clean nonzero lever arm.
    #
    # Hand computation (independent of robot.dynamics -- verified separately
    # with a standalone numpy script before writing this test):
    #   shoulder_pitch_l world pos (fixed regardless of its own angle):
    #     = (-150, 0, 300)          [from data/joints.json]
    #   R_shoulder = Rx(90) (waist_yaw = 0 so parent rotation is identity)
    #   elbow_pitch_l world pos = shoulder_pos + Rx(90) @ (0, 0, -250)
    #                            = (-150, 0, 300) + (0, 250, 0)
    #                            = (-150, 250, 300)
    #
    #   Link "elbow_pitch_l" (shoulder -> elbow), mass = 100 g = 0.1 kg:
    #     midpoint = (-150, 125, 300)
    #     r = midpoint - shoulder_pos = (0, 0.125, 0) m
    #     F = (0, 0, -0.1*9.81) = (0, 0, -0.981) N
    #     r x F = (0.125*-0.981 - 0, 0 - 0, 0 - 0) = (-0.122625, 0, 0)
    #
    #   elbow_pitch_l's own motor (GIM4310-10, 217 g = 0.217 kg), located at
    #   the elbow's own world position (downstream of the shoulder, so it
    #   loads the shoulder too):
    #     r = elbow_pos - shoulder_pos = (0, 0.25, 0) m
    #     F = (0, 0, -0.217*9.81) = (0, 0, -2.12877) N
    #     r x F = (0.25*-2.12877 - 0, 0, 0) = (-0.5321925, 0, 0)
    #
    #   shoulder_pitch_l's own motor sits at r = 0 -> contributes 0.
    #   hand_l_link mass is 0 (not provided) -> contributes 0.
    #
    #   axis_world for shoulder_pitch_l = Rx-parent(waist, angle 0) applied
    #   to local axis (1,0,0) = (1, 0, 0).
    #
    #   total = dot((-0.122625,0,0) + (-0.5321925,0,0), (1,0,0))
    #         = -0.6548175 N*m
    link_masses_g = {"elbow_pitch_l": 100.0}
    pose = {"shoulder_pitch_l": 90.0}

    torques = compute_joint_torques(config, pose, link_masses_g)

    assert torques["shoulder_pitch_l"] == pytest.approx(-0.6548175, abs=1e-6)


def test_straight_down_arm_gives_zero_shoulder_torque(config):
    # At zero pose the whole left-arm branch hangs straight along -Z from
    # the shoulder, i.e. every downstream point mass has r parallel to F
    # (both along Z), so r x F = 0 for every one of them.
    masses = estimate_all_link_masses(config)
    torques = compute_joint_torques(config, {}, masses)

    assert torques["shoulder_pitch_l"] == pytest.approx(0.0, abs=1e-9)


def test_torque_margin_flags_overload(config):
    joint_name = "elbow_pitch_l"
    spec = get_motor_spec(config.joints[joint_name].motor)  # GIM4310-10

    # Deliberately synthetic torque that exceeds the rated torque but stays
    # under stall, to check the flagging logic in both directions.
    overload_torque = spec.rated_torque_nm * 1.5
    torques_nm = {joint_name: overload_torque}

    margins = torque_margin(config, torques_nm)
    m = margins[joint_name]

    assert m["torque_nm"] == pytest.approx(overload_torque)
    assert m["rated_torque_nm"] == pytest.approx(spec.rated_torque_nm)
    assert m["stall_torque_nm"] == pytest.approx(spec.stall_torque_nm)
    assert m["pct_of_rated"] == pytest.approx(150.0)
    assert m["pct_of_rated"] > 100.0  # overload flag
    assert m["pct_of_stall"] < 100.0  # still under stall
    assert m["pct_of_stall"] == pytest.approx(overload_torque / spec.stall_torque_nm * 100.0)


def test_torque_margin_uses_absolute_value_of_signed_torque(config):
    joint_name = "shoulder_pitch_l"
    spec = get_motor_spec(config.joints[joint_name].motor)

    margins = torque_margin(config, {joint_name: -spec.rated_torque_nm})

    assert margins[joint_name]["pct_of_rated"] == pytest.approx(100.0)
