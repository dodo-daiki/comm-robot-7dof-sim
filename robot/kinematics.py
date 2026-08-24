from __future__ import annotations

import warnings

import numpy as np

from robot.joints import RobotConfig


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    kx, ky, kz = axis
    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def joint_transform(axis: np.ndarray, angle_deg: float, origin_offset_mm: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = origin_offset_mm
    R = np.eye(4)
    R[:3, :3] = _rotation_matrix(np.asarray(axis, dtype=float), angle_deg)
    return T @ R


def world_position(transform: np.ndarray) -> np.ndarray:
    return transform[:3, 3]


def _clamp_angle(name: str, angle_deg: float, range_deg: tuple[float, float]) -> float:
    lo, hi = range_deg
    if angle_deg < lo or angle_deg > hi:
        warnings.warn(
            f"Joint '{name}' angle {angle_deg} deg out of range {range_deg}; clamping."
        )
        return min(max(angle_deg, lo), hi)
    return angle_deg


def forward_kinematics(config: RobotConfig, joint_angles_deg: dict[str, float]) -> dict[str, np.ndarray]:
    transforms: dict[str, np.ndarray] = {}

    base_transform = np.eye(4)
    base_transform[2, 3] = config.base_height_mm

    def visit(joint_name: str, parent_world: np.ndarray) -> None:
        joint = config.joints[joint_name]
        angle = joint_angles_deg.get(joint_name, 0.0)
        angle = _clamp_angle(joint.name, angle, joint.range_deg)
        local = joint_transform(joint.axis, angle, joint.origin_offset_mm)
        world = parent_world @ local
        transforms[joint.name] = world

        if joint.end_effector_name is not None:
            ee_transform = np.eye(4)
            ee_transform[:3, 3] = joint.end_effector_offset_mm
            transforms[joint.end_effector_name] = world @ ee_transform

        for child in config.children(joint.name):
            visit(child.name, world)

    visit(config.root.name, base_transform)
    return transforms
