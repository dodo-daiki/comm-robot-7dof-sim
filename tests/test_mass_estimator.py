import math

import numpy as np
import pytest

from robot.joints import load_default
from robot.mass_estimator import estimate_all_link_masses, estimate_link_mass_g


def test_hollow_cylinder_mass_matches_hand_computation():
    # Hand computation (independent of robot.mass_estimator, done from the
    # formula in the module docstring):
    #   outer_r = 20, inner_r = 20 - 2.5 = 17.5
    #   outer_volume = pi * 20^2 * 100 = 125663.706... mm^3
    #   inner_core_volume = pi * 17.5^2 * 100 = 96211.275... mm^3
    #   shell_volume = outer_volume - inner_core_volume
    #   total_volume = shell_volume + 0.3 * inner_core_volume
    #   mass_g = total_volume(cm^3) * 1.27
    length_mm, diameter_mm, wall_mm, infill_ratio, density = 100.0, 40.0, 2.5, 0.3, 1.27

    outer_r, inner_r = diameter_mm / 2.0, diameter_mm / 2.0 - wall_mm
    outer_v = math.pi * outer_r**2 * length_mm
    inner_v = math.pi * inner_r**2 * length_mm
    shell_v = outer_v - inner_v
    total_v_mm3 = shell_v + infill_ratio * inner_v
    expected_g = (total_v_mm3 / 1000.0) * density

    assert expected_g == pytest.approx(74.0610833129709, rel=1e-9)

    got_g = estimate_link_mass_g(
        length_mm, diameter_mm=diameter_mm, wall_mm=wall_mm, infill_ratio=infill_ratio, density_g_cm3=density
    )
    assert got_g == pytest.approx(expected_g, rel=1e-9)


def test_mass_scales_linearly_with_length():
    base = estimate_link_mass_g(100.0, diameter_mm=40.0, wall_mm=2.5, infill_ratio=0.3)
    doubled = estimate_link_mass_g(200.0, diameter_mm=40.0, wall_mm=2.5, infill_ratio=0.3)
    tripled = estimate_link_mass_g(300.0, diameter_mm=40.0, wall_mm=2.5, infill_ratio=0.3)

    assert doubled == pytest.approx(2 * base, rel=1e-9)
    assert tripled == pytest.approx(3 * base, rel=1e-9)


def test_zero_length_gives_zero_mass():
    assert estimate_link_mass_g(0.0) == pytest.approx(0.0)


def test_degenerate_solid_rod_when_wall_exceeds_radius():
    # wall_mm (10) >= diameter_mm/2 (5) => no hollow core; clamp inner
    # radius to 0 rather than letting it go negative. Result should be a
    # simple solid cylinder, and infill_ratio must be irrelevant since there
    # is no hollow core to apply it to.
    length_mm, diameter_mm, wall_mm, density = 50.0, 10.0, 10.0, 1.27
    outer_r = diameter_mm / 2.0
    expected_g = (math.pi * outer_r**2 * length_mm / 1000.0) * density
    assert expected_g == pytest.approx(4.987278337573796, rel=1e-9)

    got_infill_0 = estimate_link_mass_g(
        length_mm, diameter_mm=diameter_mm, wall_mm=wall_mm, infill_ratio=0.0, density_g_cm3=density
    )
    got_infill_1 = estimate_link_mass_g(
        length_mm, diameter_mm=diameter_mm, wall_mm=wall_mm, infill_ratio=1.0, density_g_cm3=density
    )

    assert got_infill_0 == pytest.approx(expected_g, rel=1e-9)
    assert got_infill_1 == pytest.approx(expected_g, rel=1e-9)
    assert got_infill_0 > 0.0


def test_negative_length_raises():
    with pytest.raises(ValueError):
        estimate_link_mass_g(-1.0)


def test_infill_ratio_out_of_range_raises():
    with pytest.raises(ValueError):
        estimate_link_mass_g(100.0, infill_ratio=1.5)
    with pytest.raises(ValueError):
        estimate_link_mass_g(100.0, infill_ratio=-0.1)


def test_estimate_all_link_masses_covers_every_joint_and_end_effector():
    config = load_default()
    masses = estimate_all_link_masses(config)

    for joint_name in config.joints:
        assert joint_name in masses
    for ee_link in ("head_link", "hand_l_link", "hand_r_link"):
        assert ee_link in masses

    # waist_yaw has origin_offset_mm = [0,0,0] -> zero-length link -> ~0 mass.
    assert masses["waist_yaw"] == pytest.approx(0.0)

    # neck_yaw's link length is ||[0,0,300]|| = 300mm; must match a direct call.
    expected_neck_yaw = estimate_link_mass_g(300.0)
    assert masses["neck_yaw"] == pytest.approx(expected_neck_yaw)

    # head_link comes from neck_pitch's end_effector_offset_mm = [0,0,86]
    # (post-M9-defaults-update length; was [0,0,100]).
    expected_head_link = estimate_link_mass_g(86.0)
    assert masses["head_link"] == pytest.approx(expected_head_link)


def test_estimate_all_link_masses_forwards_overrides():
    config = load_default()
    default_masses = estimate_all_link_masses(config)
    denser_masses = estimate_all_link_masses(config, diameter_mm=60.0)

    # A larger diameter must increase mass for every link with nonzero length.
    assert denser_masses["neck_yaw"] > default_masses["neck_yaw"]
