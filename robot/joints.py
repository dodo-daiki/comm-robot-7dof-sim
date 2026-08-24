from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_JOINTS_JSON_PATH = Path(__file__).resolve().parent.parent / "data" / "joints.json"


@dataclass
class Joint:
    name: str
    parent: str | None
    part_ja: str
    axis: np.ndarray
    origin_offset_mm: np.ndarray
    range_deg: tuple[float, float]
    motor: str
    gear_ratio: float
    can_id: str
    end_effector_name: str | None = None
    end_effector_offset_mm: np.ndarray | None = None


class RobotConfig:
    def __init__(self, data: dict):
        self.coordinate_frame: str = data["coordinate_frame"]
        self.base_height_mm: float = data["base_height_mm"]
        self.joints: dict[str, Joint] = {}
        self.root: Joint | None = None

        roots: list[Joint] = []
        for entry in data["joints"]:
            end_effector_name = entry.get("end_effector_name")
            end_effector_offset = entry.get("end_effector_offset_mm")
            if (end_effector_name is None) != (end_effector_offset is None):
                raise ValueError(
                    f"Joint '{entry['name']}' must set both 'end_effector_name' and "
                    "'end_effector_offset_mm' together, or neither."
                )
            joint = Joint(
                name=entry["name"],
                parent=entry["parent"],
                part_ja=entry["part_ja"],
                axis=np.array(entry["axis"], dtype=float),
                origin_offset_mm=np.array(entry["origin_offset_mm"], dtype=float),
                range_deg=tuple(entry["range_deg"]),
                motor=entry["motor"],
                gear_ratio=entry["gear_ratio"],
                can_id=entry["can_id"],
                end_effector_name=end_effector_name,
                end_effector_offset_mm=(
                    np.array(end_effector_offset, dtype=float)
                    if end_effector_offset is not None
                    else None
                ),
            )
            self.joints[joint.name] = joint
            if joint.parent is None:
                roots.append(joint)

        for joint in self.joints.values():
            if joint.parent is not None and joint.parent not in self.joints:
                raise ValueError(
                    f"Joint '{joint.name}' references unknown parent '{joint.parent}'."
                )

        if len(roots) != 1:
            raise ValueError(
                f"RobotConfig requires exactly one root joint (parent=null); found {len(roots)}: "
                f"{[j.name for j in roots]}"
            )
        self.root = roots[0]

    def children(self, name: str) -> list[Joint]:
        return [j for j in self.joints.values() if j.parent == name]


def load_default(path: Path | str | None = None) -> RobotConfig:
    json_path = Path(path) if path is not None else DEFAULT_JOINTS_JSON_PATH
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RobotConfig(data)
