"""Estimate link masses assuming links are 3D-printed in PETG.

Real link masses aren't finalized yet (the mechanical design isn't done), so
until then every link is modeled as a simple hollow cylinder -- outer
diameter, wall thickness, and a partial-infill core -- printed in PETG.  This
is a coarse stand-in for a real slicer estimate, not a precision model.

All parameters are exposed as keyword arguments with sensible defaults so a
future UI (Milestone 3) can override them per-call without touching this
module.
"""

from __future__ import annotations

import math

import numpy as np

from robot.joints import RobotConfig

# PETG density in g/cm^3 (typical published value for PETG filament).
PETG_DENSITY_G_CM3 = 1.27

# Sensible defaults for an as-yet-unfinalized mechanical design.
DEFAULT_DIAMETER_MM = 40.0
DEFAULT_WALL_MM = 2.5
DEFAULT_INFILL_RATIO = 0.3


def estimate_link_mass_g(
    length_mm: float,
    diameter_mm: float = DEFAULT_DIAMETER_MM,
    wall_mm: float = DEFAULT_WALL_MM,
    infill_ratio: float = DEFAULT_INFILL_RATIO,
    density_g_cm3: float = PETG_DENSITY_G_CM3,
) -> float:
    """Estimate the mass (grams) of a link modeled as a hollow cylinder.

    The cylinder has outer diameter `diameter_mm`, wall thickness `wall_mm`,
    and length `length_mm`. The wall itself is treated as fully solid
    (100% infill); the hollow core inside the wall is treated as printed at
    `infill_ratio` density (0 = empty, 1 = fully solid).

    mass = density * (shell_volume + infill_ratio * inner_core_volume)

    Degenerate case: if `wall_mm >= diameter_mm / 2`, there is no hollow
    core -- the wall thickness consumes the entire cross-section. Rather
    than raise, this is treated as a solid rod: the inner core radius is
    clamped to 0 (never negative), so `inner_core_volume` becomes 0 and the
    "shell" is simply the full solid cylinder. `infill_ratio` is irrelevant
    in that case since there is no hollow core left to apply it to.
    """
    if length_mm < 0:
        raise ValueError(f"length_mm must be >= 0, got {length_mm}")
    if diameter_mm <= 0:
        raise ValueError(f"diameter_mm must be > 0, got {diameter_mm}")
    if wall_mm < 0:
        raise ValueError(f"wall_mm must be >= 0, got {wall_mm}")
    if not 0.0 <= infill_ratio <= 1.0:
        raise ValueError(f"infill_ratio must be in [0, 1], got {infill_ratio}")

    outer_radius_mm = diameter_mm / 2.0
    inner_radius_mm = max(outer_radius_mm - wall_mm, 0.0)  # clamp: solid rod, never negative

    outer_volume_mm3 = math.pi * outer_radius_mm**2 * length_mm
    inner_core_volume_mm3 = math.pi * inner_radius_mm**2 * length_mm
    shell_volume_mm3 = outer_volume_mm3 - inner_core_volume_mm3

    total_volume_mm3 = shell_volume_mm3 + infill_ratio * inner_core_volume_mm3
    total_volume_cm3 = total_volume_mm3 / 1000.0  # 1 cm^3 = 1000 mm^3

    return total_volume_cm3 * density_g_cm3


def estimate_all_link_masses(config: RobotConfig, **overrides) -> dict[str, float]:
    """Estimate a mass (grams) for every link in the robot.

    Returns one entry per joint, keyed by the joint's own name, representing
    the link running from that joint's parent into the joint (length =
    ||joint.origin_offset_mm||). Additionally, for every joint that carries
    an end effector, an extra "<end_effector_name>_link" entry is added for
    the terminal segment (length = ||joint.end_effector_offset_mm||), e.g.
    "head_link", "hand_l_link", "hand_r_link".

    `**overrides` are forwarded to `estimate_link_mass_g` for every link
    (e.g. diameter_mm=35.0), letting a caller (future UI) override the
    printing parameters uniformly for all links in one call.
    """
    masses: dict[str, float] = {}

    for joint in config.joints.values():
        length_mm = float(np.linalg.norm(joint.origin_offset_mm))
        masses[joint.name] = estimate_link_mass_g(length_mm, **overrides)

        if joint.end_effector_name is not None:
            ee_length_mm = float(np.linalg.norm(joint.end_effector_offset_mm))
            masses[f"{joint.end_effector_name}_link"] = estimate_link_mass_g(
                ee_length_mm, **overrides
            )

    return masses
