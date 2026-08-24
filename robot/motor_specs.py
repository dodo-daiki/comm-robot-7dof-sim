"""Motor + gearbox specs for the actuators used in this robot design.

These figures are transcribed from public product/datasheet pages, not from
the manufacturer's original PDF datasheet (steadywin GIM4310 series). They
should be CONFIRMED against the official datasheet before being relied on
for real hardware sizing -- treat them as engineering estimates for
Milestone 2's torque-margin checks, not as verified specs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorSpec:
    name: str
    rated_torque_nm: float
    stall_torque_nm: float
    rated_speed_rpm: float
    weight_g: float
    source_url: str


# GIM4310-10: 10:1 planetary-geared version of the steadywin GIM4310 servo motor.
# Source: https://aifitlab.com/products/steadywin-gim4310-10-planetary-reducer-servo-motor
# NOTE: confirm against the official steadywin datasheet before real use.
GIM4310_10 = MotorSpec(
    name="GIM4310-10",
    rated_torque_nm=2.05,
    stall_torque_nm=5.6,
    rated_speed_rpm=150,
    weight_g=217,
    source_url="https://aifitlab.com/products/steadywin-gim4310-10-planetary-reducer-servo-motor",
)

# GIM4310-36: 36:1 planetary-geared version of the steadywin GIM4310 servo motor.
# Source: https://lucidar.me/en/actuators/mit-motor-gim4310-36/
# NOTE: confirm against the official steadywin datasheet before real use.
GIM4310_36 = MotorSpec(
    name="GIM4310-36",
    rated_torque_nm=7.38,
    stall_torque_nm=20.16,
    rated_speed_rpm=41,
    weight_g=310,
    source_url="https://lucidar.me/en/actuators/mit-motor-gim4310-36/",
)

_MOTOR_SPECS: dict[str, MotorSpec] = {
    GIM4310_10.name: GIM4310_10,
    GIM4310_36.name: GIM4310_36,
}


def get_motor_spec(name: str) -> MotorSpec:
    """Look up a MotorSpec by name (e.g. a Joint.motor string like 'GIM4310-10')."""
    try:
        return _MOTOR_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_MOTOR_SPECS))
        raise KeyError(f"Unknown motor '{name}'. Known motors: {known}") from exc
