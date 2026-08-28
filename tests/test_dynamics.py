import numpy as np
import pytest

from robot.dynamics import compute_joint_torques, torque_margin
from robot.joints import load_default
from robot.mass_estimator import estimate_all_link_masses
from robot.motor_specs import get_motor_spec


@pytest.fixture
def config():
    return load_default()


def test_yaw_joint_gravity_torque_is_zero_in_this_tree(config):
    # NOTE: this is a property of THIS tree's topology, not a general law of
    # the dynamics model. A yaw (Z-axis) joint gets zero gravity torque only
    # when its world-frame axis is exactly (0, 0, 1); a yaw joint sitting
    # under a PITCH ancestor would generally have a tilted world axis and
    # would pick up nonzero gravity torque. Here it happens that:
    #   - waist_yaw has no parent (base frame has no rotation), and
    #   - neck_yaw's only ancestor is waist_yaw, which also rotates about Z
    #     -- and a Z-rotation leaves the vector (0,0,1) unchanged,
    # so both keep an exact world axis of (0,0,1). Since gravity F=(0,0,-mg)
    # always has Fx=Fy=0, the Z component of any r x F (r_x*F_y - r_y*F_x)
    # is then always exactly 0, regardless of r. That is what's checked
    # below, for this specific config -- not asserted as a general rule.
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
    # with a standalone numpy script before writing this test), using the
    # post-M9-defaults-update elbow_pitch_l origin offset of (0, 0, -120)
    # (was (0, 0, -250)):
    #   shoulder_pitch_l world pos (fixed regardless of its own angle):
    #     = (0, 0, base_height_mm) + (-125.21980673998821, 0, 250.43961347997643)
    #     [base_height_mm = 500, offset from data/joints.json]
    #   R_shoulder = Rx(90) (waist_yaw = 0 so parent rotation is identity)
    #   elbow_pitch_l world pos = shoulder_pos + Rx(90) @ (0, 0, -120)
    #                            = shoulder_pos + (0, 120, 0)
    #   (only the relative lever-arm vectors below actually feed the
    #   assertion, so the shared shoulder-pos offset drops out of every r;
    #   it's spelled out here only so the comment matches the real geometry.)
    #
    #   Link "elbow_pitch_l" (shoulder -> elbow), mass = 100 g = 0.1 kg:
    #     r = midpoint - shoulder_pos = (0, 0.06, 0) m
    #     F = (0, 0, -0.1*9.81) = (0, 0, -0.981) N
    #     r x F = (0.06*-0.981 - 0, 0 - 0, 0 - 0) = (-0.05886, 0, 0)
    #
    #   elbow_pitch_l's own motor (GIM4310-10, 217 g = 0.217 kg), located at
    #   the elbow's own world position (downstream of the shoulder, so it
    #   loads the shoulder too):
    #     r = elbow_pos - shoulder_pos = (0, 0.12, 0) m
    #     F = (0, 0, -0.217*9.81) = (0, 0, -2.12877) N
    #     r x F = (0.12*-2.12877 - 0, 0, 0) = (-0.2554524, 0, 0)
    #
    #   shoulder_pitch_l's own motor sits at r = 0 -> contributes 0.
    #   hand_l_link mass is 0 (not provided) -> contributes 0.
    #
    #   axis_world for shoulder_pitch_l = Rx-parent(waist, angle 0) applied
    #   to local axis (1,0,0) = (1, 0, 0).
    #
    #   total = dot((-0.05886,0,0) + (-0.2554524,0,0), (1,0,0))
    #         = -0.3143124 N*m
    link_masses_g = {"elbow_pitch_l": 100.0}
    pose = {"shoulder_pitch_l": 90.0}

    torques = compute_joint_torques(config, pose, link_masses_g)

    assert torques["shoulder_pitch_l"] == pytest.approx(-0.3143124, abs=1e-6)


def test_straight_down_arm_gives_zero_shoulder_torque(config):
    # At zero pose the whole left-arm branch hangs straight along -Z from
    # the shoulder, i.e. every downstream point mass has r parallel to F
    # (both along Z), so r x F = 0 for every one of them.
    masses = estimate_all_link_masses(config)
    torques = compute_joint_torques(config, {}, masses)

    assert torques["shoulder_pitch_l"] == pytest.approx(0.0, abs=1e-9)


def test_shoulder_torque_with_nonzero_waist_yaw_exercises_world_axis_projection(config):
    # Regression test for the highest-risk area flagged in the M2 brief: a
    # joint's rotation axis must be projected into world frame via its
    # PARENT's accumulated world rotation (parent_rotation @ local_axis),
    # not used as a raw local vector. No other existing test combines a
    # nonzero waist_yaw with a nonzero downstream pitch joint, so a bug that
    # returns the raw local axis instead of the world-projected one was not
    # previously caught (verified: mutating _joint_axis_world to return the
    # raw local axis makes this assertion fail with a ~29% error, per an
    # external review that reproduced this independently).
    #
    # Pose: waist_yaw=45, shoulder_pitch_l=90, default (estimate_all_link_masses)
    # masses. Hand computation, done from scratch with a standalone numpy
    # script (not by calling robot.dynamics) before encoding this assertion,
    # using the post-M9-defaults-update offsets (shoulder_pitch_l origin
    # offset now (-125.21980673998821, 0, 250.43961347997643), elbow_pitch_l
    # origin/end-effector offsets now both (0, 0, -120)):
    #   waist_pos = (0, 0, 500); R_waist = Rz(45)
    #   shoulder_pos = waist_pos + Rz(45) @ (-125.21980673998821, 0, 250.43961347997643)
    #                = (-88.54377448471462, -88.54377448471462, 750.4396134799764)
    #   R_shoulder = Rz(45) @ Rx(90)
    #   elbow_pos = shoulder_pos + R_shoulder @ (0, 0, -120)
    #             = (-173.39658822710032, -3.6909607423289117, 750.4396134799764)
    #   R_elbow = R_shoulder (elbow_pitch_l = 0)
    #   hand_l_pos = elbow_pos + R_elbow @ (0, 0, -120)
    #              = (-258.24940196948603, 81.1618530000568, 750.4396134799764)
    #
    #   axis_world(shoulder_pitch_l) = R_waist @ (1,0,0) = (cos45, sin45, 0)
    #     -- THIS is the projection under test: if the axis were left as the
    #     raw local (1,0,0) instead, the dot products below would use the
    #     wrong direction and give a materially different (wrong) answer.
    #
    #   Downstream of shoulder_pitch_l: link "elbow_pitch_l" (shoulder->elbow
    #   midpoint, length 120mm, mass ~88.873 g), link "hand_l_link"
    #   (elbow->hand_l midpoint, length 120mm, mass ~88.873 g -- the two are
    #   now the same length, hence the same mass), and elbow_pitch_l's own
    #   motor (GIM4310-10, 217 g) at elbow_pos. shoulder_pitch_l's own motor
    #   sits at r=0. Summing torque_about(shoulder_pos, axis_world, point,
    #   mass) over those three contributions gives:
    #     t_elbow_link  = -0.0523108243656176
    #     t_hand_link   = -0.1569324730968528
    #     t_elbow_motor = -0.2554524000000001
    #     total = -0.46469569746247047 N*m
    masses = estimate_all_link_masses(config)
    pose = {"waist_yaw": 45.0, "shoulder_pitch_l": 90.0}

    torques = compute_joint_torques(config, pose, masses)

    assert torques["shoulder_pitch_l"] == pytest.approx(-0.46469569746247047, abs=1e-6)


def test_elbow_pitch_l_torque_matches_hand_computation(config):
    # Downstream of elbow_pitch_l there is only its own end-effector link
    # (hand_l_link) -- its own motor sits at r=0. Pose shoulder_pitch_l=90
    # swings the forearm to +Y, perpendicular to gravity, giving a clean
    # nonzero lever arm about elbow_pitch_l's own axis.
    #
    # Hand computation (standalone numpy script, independent of
    # robot.dynamics), using the post-M9-defaults-update elbow_pitch_l
    # origin/end-effector offsets of (0, 0, -120) each (was -250/-230):
    #   shoulder_pos = (-125.21980673998821, 0, 750.4396134799764); R_shoulder = Rx(90)
    #   elbow_pos = shoulder_pos + Rx(90) @ (0,0,-120) = shoulder_pos + (0, 120, 0)
    #   R_elbow = R_shoulder (elbow_pitch_l angle = 0)
    #   hand_l_pos = elbow_pos + R_elbow @ (0,0,-120) = elbow_pos + (0, 120, 0)
    #   midpoint = (elbow_pos + hand_l_pos) / 2; r = midpoint - elbow_pos
    #            = (0, 60, 0) mm = (0, 0.06, 0) m
    #   axis_world(elbow_pitch_l) = R_shoulder @ (1,0,0) = (1,0,0)
    #     (Rx leaves the X axis itself unchanged, so this stays (1,0,0)
    #     regardless of the shoulder angle -- unlike the shoulder's own axis
    #     in the waist-yaw test above.)
    #   mass(hand_l_link) [defaults, length 120mm] = 88.87329997556506 g = 0.08887... kg
    #   F = (0, 0, -0.08887329997556506*9.81) N
    #   r x F = (0.06 * F_z, 0, 0); dot with (1,0,0) = 0.06 * F_z
    #         = -0.0523108243656176 N*m
    masses = estimate_all_link_masses(config)
    torques = compute_joint_torques(config, {"shoulder_pitch_l": 90.0}, masses)

    assert torques["elbow_pitch_l"] == pytest.approx(-0.0523108243656176, abs=1e-6)


def test_elbow_pitch_r_torque_matches_mirrored_hand_computation(config):
    # Mirror of the elbow_pitch_l case above: shoulder_pitch_r's mount is at
    # x=+150 instead of x=-150, but that only shifts a constant that cancels
    # out of every r vector, so with the same-sign angle (per this design's
    # pitch-only, no-roll convention -- see test_kinematics.py) the resulting
    # torque about elbow_pitch_r's own axis is numerically identical to the
    # left-arm case.
    masses = estimate_all_link_masses(config)
    torques = compute_joint_torques(config, {"shoulder_pitch_r": 90.0}, masses)

    assert torques["elbow_pitch_r"] == pytest.approx(-0.0523108243656176, abs=1e-6)


def test_neck_pitch_torque_matches_hand_computation(config):
    # Downstream of neck_pitch there is only its own end-effector link
    # (head_link) -- its own motor sits at r=0. Pose neck_pitch=45 (its max
    # range_deg, [-30, 45] per data/joints.json -- 90 would be silently
    # clamped by forward_kinematics) swings the head stub off-axis, giving a
    # nonzero lever arm.
    #
    # Hand computation (standalone numpy script, independent of
    # robot.dynamics), using the post-M9-defaults-update neck_yaw->neck_pitch
    # offset of (0,0,25) (was (0,0,60)) and neck_pitch->head offset of
    # (0,0,86) (was (0,0,100)):
    #   neck_yaw_pos = (0, 0, 800); R_neck_yaw = I (neck_yaw angle = 0)
    #   neck_pitch_pos = neck_yaw_pos + I @ (0,0,25) = (0, 0, 825)
    #   R_neck_pitch = R_neck_yaw @ Rx(45) = Rx(45)
    #   head_pos = neck_pitch_pos + Rx(45) @ (0,0,86)
    #            = (0, -60.81118318204309, 885.8111831820431)
    #   midpoint = (neck_pitch_pos + head_pos) / 2
    #   r = midpoint - neck_pitch_pos = (0, -30.405591591021543, 30.40559159102156) mm
    #   axis_world(neck_pitch) = R_neck_yaw @ (1,0,0) = (1,0,0)
    #   mass(head_link) [defaults, length 86mm] = 63.69253164915496 g = 0.06369... kg
    #   F = (0, 0, -0.06369253164915496*9.81) N
    #   torque_about = dot(r/1000 x F, (1,0,0)) = 0.01899813531732694 N*m
    masses = estimate_all_link_masses(config)
    torques = compute_joint_torques(config, {"neck_pitch": 45.0}, masses)

    assert torques["neck_pitch"] == pytest.approx(0.01899813531732694, abs=1e-6)


def test_link_mass_loads_parent_joint_not_itself(config):
    # Regression test for a link-anchor off-by-one: the "elbow_pitch_l" link
    # (the rod running from shoulder_pitch_l to elbow_pitch_l) must load its
    # PARENT (shoulder_pitch_l), not itself. If the anchor were mistakenly
    # set to the link's own joint name instead of joint.parent, this link's
    # mass would (wrongly) load elbow_pitch_l and NOT shoulder_pitch_l.
    pose = {"shoulder_pitch_l": 90.0}
    light = compute_joint_torques(config, pose, {"elbow_pitch_l": 50.0})
    heavy = compute_joint_torques(config, pose, {"elbow_pitch_l": 500.0})

    # Changing this link's mass must change the parent shoulder's torque...
    assert heavy["shoulder_pitch_l"] != pytest.approx(light["shoulder_pitch_l"])
    # ...but must NOT change elbow_pitch_l's own torque, since this link
    # does not hang below elbow_pitch_l -- it hangs below shoulder_pitch_l.
    assert heavy["elbow_pitch_l"] == pytest.approx(light["elbow_pitch_l"])


def test_payload_mass_adds_point_load_at_end_effector(config):
    # payload_mass_g places an extra point mass at the end effector's own
    # (non-midpoint) world position, e.g. a head assembly not otherwise
    # captured by the printed neck-stub link mass. Pose: neck_pitch=45 (its
    # max range_deg -- 90 would be silently clamped) puts head off-axis from
    # neck_pitch, same geometry as
    # test_neck_pitch_torque_matches_hand_computation above.
    #
    # Hand computation (standalone numpy script, independent of
    # robot.dynamics), using the post-M9-defaults-update offsets (see
    # test_neck_pitch_torque_matches_hand_computation above):
    #   neck_pitch_pos = (0, 0, 825)
    #   head_pos = (0, -60.81118318204309, 885.8111831820431)   [Rx(45) @ (0,0,86)]
    #   axis_world(neck_pitch) = (1, 0, 0)
    #   link-only contribution (head_link mass = 63.69253164915496 g) =
    #     0.01899813531732694 N*m [see test above]
    #   payload contribution (500 g at head_pos exactly,
    #     r = head_pos - neck_pitch_pos = (0, -0.06081118318204309, 0.06081118318204309) m):
    #     F = (0, 0, -0.5*9.81) = (0, 0, -4.905)
    #     torque_about = dot(r x F, (1,0,0)) = 0.2982788535079214 N*m
    #   total = 0.01899813531732694 + 0.2982788535079214
    #         = 0.3172769888252483 N*m
    masses = estimate_all_link_masses(config)
    pose = {"neck_pitch": 45.0}

    without_payload = compute_joint_torques(config, pose, masses)
    with_payload = compute_joint_torques(config, pose, masses, payload_mass_g={"head": 500.0})

    assert without_payload["neck_pitch"] == pytest.approx(0.01899813531732694, abs=1e-6)
    assert with_payload["neck_pitch"] == pytest.approx(0.3172769888252483, abs=1e-6)

    # A payload on a different end effector must not affect neck_pitch.
    hand_payload_only = compute_joint_torques(config, pose, masses, payload_mass_g={"hand_l": 500.0})
    assert hand_payload_only["neck_pitch"] == pytest.approx(without_payload["neck_pitch"])


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
