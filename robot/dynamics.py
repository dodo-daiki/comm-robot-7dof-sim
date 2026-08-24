"""Static gravity-torque estimate for a given robot pose.

This is deliberately NOT full rigid-body dynamics (no inertia, no
acceleration/velocity terms, no friction) -- just a static free-body
estimate of how much torque gravity alone puts on each joint, given a pose
and a set of link masses. It is meant to sanity-check motor selection
(`motor_specs.py`) against link-mass estimates (`mass_estimator.py`), not to
drive a controller.

World transforms and world positions are obtained from M1's
`robot.kinematics.forward_kinematics` / `world_position` so this module
never re-derives rotations independently of M1 -- it only reads the
resulting transform matrices.

TODO(M3 UI): `compute_joint_torques`'s `payload_mass_g` lets a caller add an
extra point mass at an end effector (head assembly, gripper payload, etc.)
beyond what the printed-link mass estimate captures. It defaults to no
payload for every end effector, which is why out-of-the-box torque margins
are optimistic (e.g. "head" currently only carries its ~74 g neck stub, not
a real head/face assembly). Wire this up to a per-end-effector UI slider
once Milestone 3's app exists, so users can explore that margin directly.
"""

from __future__ import annotations

import numpy as np

from robot.joints import Joint, RobotConfig
from robot.kinematics import forward_kinematics, world_position
from robot.motor_specs import get_motor_spec

GRAVITY_M_S2 = 9.81


def _base_transform(config: RobotConfig) -> np.ndarray:
    base = np.eye(4)
    base[2, 3] = config.base_height_mm
    return base


def _ancestor_chain_inclusive(config: RobotConfig, joint_name: str) -> set[str]:
    """Names of `joint_name` and all of its ancestors, up to the root."""
    chain: set[str] = set()
    name: str | None = joint_name
    while name is not None:
        chain.add(name)
        name = config.joints[name].parent
    return chain


def _joint_axis_world(
    config: RobotConfig,
    joint: Joint,
    transforms: dict[str, np.ndarray],
    base_transform: np.ndarray,
) -> np.ndarray:
    """World-frame direction of `joint`'s own rotation axis.

    A joint's axis is defined in its parent's world-orientation frame (the
    frame in effect after the parent's own rotation but before this joint's
    own rotation is applied) -- this mirrors exactly how
    `robot.kinematics.joint_transform` uses `joint.axis` inside
    `parent_world @ local`.
    """
    parent_rotation = (
        transforms[joint.parent][:3, :3] if joint.parent is not None else base_transform[:3, :3]
    )
    axis = joint.axis / np.linalg.norm(joint.axis)
    return parent_rotation @ axis


def compute_joint_torques(
    config: RobotConfig,
    joint_angles_deg: dict[str, float],
    link_masses_g: dict[str, float],
    payload_mass_g: dict[str, float] | None = None,
) -> dict[str, float]:
    """Static gravitational torque (N*m) about each joint's own rotation axis.

    For every joint Q, this sums, over every link and every motor mass that
    is downstream of Q (Q itself or a proper ancestor of it in the tree),
    the torque about Q's world-frame rotation axis caused by that mass's
    weight acting at its world position:

        torque_vector = r x F
        F = [0, 0, -mass_kg * 9.81]
        r = mass_world_position - Q_world_position
        torque_about_Q = dot(torque_vector, Q_axis_world)

    Links: each joint J owns the physical link running from J.parent's
    world position to J's own world position (mass = link_masses_g[J.name]),
    modeled as a point mass at the link's midpoint. A joint carrying an end
    effector additionally owns a terminal link from its own world position
    to the end effector's world position (mass =
    link_masses_g["<end_effector_name>_link"]). A link is downstream of Q
    iff Q is in the ancestor-inclusive chain of the link's *anchor* joint
    (J.parent for a normal link, J itself for an end-effector link) --
    i.e. iff rotating Q would move that link. Missing masses default to 0.

    Motors: each joint D's own motor+gearbox mass (from `motor_specs`) is
    treated as a point mass at D's own world position, and is downstream of
    Q iff Q is in D's ancestor-inclusive chain (Q == D or Q is an ancestor
    of D) -- e.g. the elbow motor's weight loads the shoulder. A motor at
    its own joint contributes zero torque to that same joint (zero moment
    arm), so including D == Q is harmless.

    Payload (optional): `payload_mass_g` maps an end-effector name (e.g.
    "head", "hand_l", "hand_r") to an extra point mass in grams, placed at
    that end effector's own world position (not the link midpoint) --
    approximating a head assembly, gripper, or held object not otherwise
    captured by the printed-link mass estimate. Governed by the same
    ancestor rule as that end effector's link. Defaults to no payload for
    every end effector.
    """
    payload_mass_g = payload_mass_g or {}
    transforms = forward_kinematics(config, joint_angles_deg)
    base_transform = _base_transform(config)

    def joint_world_pos(name: str) -> np.ndarray:
        return world_position(transforms[name])

    # (anchor_joint_name, point_world_mm, mass_kg) for every link and payload.
    link_loads: list[tuple[str, np.ndarray, float]] = []
    for joint in config.joints.values():
        parent_pos = (
            joint_world_pos(joint.parent) if joint.parent is not None else world_position(base_transform)
        )
        this_pos = joint_world_pos(joint.name)
        mass_kg = link_masses_g.get(joint.name, 0.0) / 1000.0
        if joint.parent is not None:  # root's own "link into it" has no governing joint
            link_loads.append((joint.parent, (parent_pos + this_pos) / 2.0, mass_kg))

        if joint.end_effector_name is not None:
            ee_pos = joint_world_pos(joint.end_effector_name)
            ee_mass_kg = link_masses_g.get(f"{joint.end_effector_name}_link", 0.0) / 1000.0
            link_loads.append((joint.name, (this_pos + ee_pos) / 2.0, ee_mass_kg))

            payload_kg = payload_mass_g.get(joint.end_effector_name, 0.0) / 1000.0
            if payload_kg:
                link_loads.append((joint.name, ee_pos, payload_kg))

    # (owning_joint_name, motor_world_mm, mass_kg) for every motor+gearbox.
    motor_loads: list[tuple[str, np.ndarray, float]] = []
    for joint in config.joints.values():
        motor_mass_kg = get_motor_spec(joint.motor).weight_g / 1000.0
        motor_loads.append((joint.name, joint_world_pos(joint.name), motor_mass_kg))

    torques: dict[str, float] = {}
    for joint in config.joints.values():
        q_pos = joint_world_pos(joint.name)
        axis_world = _joint_axis_world(config, joint, transforms, base_transform)

        total_torque = 0.0
        for anchor_name, midpoint_mm, mass_kg in link_loads:
            if joint.name not in _ancestor_chain_inclusive(config, anchor_name):
                continue
            r_m = (midpoint_mm - q_pos) / 1000.0
            force = np.array([0.0, 0.0, -mass_kg * GRAVITY_M_S2])
            total_torque += float(np.dot(np.cross(r_m, force), axis_world))

        for owner_name, motor_pos_mm, mass_kg in motor_loads:
            if joint.name not in _ancestor_chain_inclusive(config, owner_name):
                continue
            r_m = (motor_pos_mm - q_pos) / 1000.0
            force = np.array([0.0, 0.0, -mass_kg * GRAVITY_M_S2])
            total_torque += float(np.dot(np.cross(r_m, force), axis_world))

        torques[joint.name] = total_torque

    return torques


def torque_margin(config: RobotConfig, torques_nm: dict[str, float]) -> dict[str, dict]:
    """For each joint, compare its computed torque against its motor's ratings.

    Returns joint_name -> {torque_nm, rated_torque_nm, stall_torque_nm,
    pct_of_rated, pct_of_stall}. Percentages are computed against the
    magnitude of the torque (a joint can be loaded in either rotation
    direction).
    """
    margins: dict[str, dict] = {}
    for joint_name, torque_nm in torques_nm.items():
        spec = get_motor_spec(config.joints[joint_name].motor)
        abs_torque = abs(torque_nm)
        margins[joint_name] = {
            "torque_nm": torque_nm,
            "rated_torque_nm": spec.rated_torque_nm,
            "stall_torque_nm": spec.stall_torque_nm,
            "pct_of_rated": abs_torque / spec.rated_torque_nm * 100.0,
            "pct_of_stall": abs_torque / spec.stall_torque_nm * 100.0,
        }
    return margins
