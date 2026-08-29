"""Minimal independent numerical validation for the LIPM/DCM certificate."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from legged_lab.recovery import (
    CertificateState,
    RecoverabilityConfig,
    certify_recoverability,
    check_witness,
    terminal_contains,
)
from legged_lab.recovery.certificate import propagate_dcm


@dataclass(frozen=True)
class NumericalLIPMState:
    position: np.ndarray
    velocity: np.ndarray


def integrate_lipm_rk4(
    state: NumericalLIPMState,
    cop: np.ndarray,
    omega: float,
    duration: float,
    max_dt: float = 5.0e-4,
) -> NumericalLIPMState:
    """Integrate ``c_ddot = omega^2 (c - r)`` without using DCM propagation."""
    if duration < 0.0 or max_dt <= 0.0:
        raise ValueError("duration must be non-negative and max_dt must be positive.")
    if duration == 0.0:
        return state

    num_steps = max(1, math.ceil(duration / max_dt))
    dt = duration / num_steps
    value = np.r_[state.position, state.velocity].astype(np.float64)
    cop = np.asarray(cop, dtype=np.float64)

    def derivative(vector: np.ndarray) -> np.ndarray:
        position = vector[:2]
        velocity = vector[2:]
        return np.r_[velocity, omega * omega * (position - cop)]

    for _ in range(num_steps):
        k1 = derivative(value)
        k2 = derivative(value + 0.5 * dt * k1)
        k3 = derivative(value + 0.5 * dt * k2)
        k4 = derivative(value + dt * k3)
        value += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return NumericalLIPMState(position=value[:2], velocity=value[2:])


def dcm_offset(state: NumericalLIPMState, support_position: np.ndarray, omega: float) -> np.ndarray:
    xi = state.position + state.velocity / omega
    return xi - support_position


def numerical_touchdown_step(
    state: NumericalLIPMState,
    support_position: np.ndarray,
    cop_offset: np.ndarray,
    landing: np.ndarray,
    omega: float,
    duration: float,
) -> tuple[NumericalLIPMState, np.ndarray, np.ndarray]:
    """Integrate one support phase and then switch the support point."""
    cop_world = support_position + cop_offset
    integrated = integrate_lipm_rk4(state, cop_world, omega, duration)
    new_support = support_position + landing
    return integrated, new_support, dcm_offset(integrated, new_support, omega)


def state_from_dcm(b: np.ndarray, support_position: np.ndarray, omega: float) -> NumericalLIPMState:
    """Create a non-trivial c/c_dot pair with the requested DCM offset."""
    xi = support_position + b
    position = support_position + 0.37 * b
    velocity = omega * (xi - position)
    return NumericalLIPMState(position=position, velocity=velocity)


def artificial_states(config: RecoverabilityConfig) -> dict[int, CertificateState]:
    """Deterministic states chosen to exercise exact horizons 3, 5, and >5."""
    nominal_b, nominal_q = config.terminal_nominal("left")
    q = nominal_q + np.array((-0.12, -0.08))
    return {
        3: CertificateState(nominal_b + np.array((-0.1450, 0.0)), q, "left", phase=0.0),
        5: CertificateState(nominal_b + np.array((-0.1475, 0.0)), q, "left", phase=0.0),
        6: CertificateState(nominal_b + np.array((-0.2000, 0.0)), q, "left", phase=0.0),
    }


def validate_artificial_horizons(config: RecoverabilityConfig):
    results = {}
    max_residual = 0.0
    for expected, state in artificial_states(config).items():
        result = certify_recoverability(state, config)
        if result.n_min != expected:
            raise AssertionError(f"Expected N_min={expected}, got {result.n_min}: {result.message}")
        results[expected] = result
        if expected <= 5:
            if result.witness is None:
                raise AssertionError(f"N_min={expected} did not return a witness.")
            residual = check_witness(state, result.witness, config)
            max_residual = max(
                max_residual,
                residual.max_equality,
                residual.max_inequality_violation,
                residual.max_dynamics,
                residual.max_support_switch,
            )
    if max_residual > 1.0e-7:
        raise AssertionError(f"LP witness residual is too large: {max_residual:.3e}")
    return results, max_residual


def validate_single_step(config: RecoverabilityConfig, rng: np.random.Generator) -> float:
    omega = config.omega
    support = rng.uniform((-0.2, -0.2), (0.2, 0.2))
    position = support + rng.uniform((-0.12, -0.08), (0.12, 0.08))
    velocity = rng.uniform((-0.35, -0.25), (0.35, 0.25))
    cop = rng.uniform((-0.05, -0.025), (0.08, 0.025))
    landing = rng.uniform((-0.10, -0.32), (0.45, -0.14))
    duration = 0.43

    initial = NumericalLIPMState(position, velocity)
    b0 = dcm_offset(initial, support, omega)
    theory = propagate_dcm(b0, cop, landing, omega, duration)
    _, _, numerical = numerical_touchdown_step(initial, support, cop, landing, omega, duration)
    return float(np.max(np.abs(theory - numerical)))


def validate_phase(config: RecoverabilityConfig) -> float:
    omega = config.omega
    support = np.array((0.12, -0.08))
    initial = NumericalLIPMState(np.array((0.18, -0.03)), np.array((0.22, -0.11)))
    cop = np.array((0.025, -0.012))
    landing = np.array((0.31, -0.21))
    b0 = dcm_offset(initial, support, omega)
    errors = []
    for phase in (0.3, 0.7):
        remaining = (1.0 - phase) * config.step_period
        theory = propagate_dcm(b0, cop, landing, omega, remaining)
        _, _, numerical = numerical_touchdown_step(initial, support, cop, landing, omega, remaining)
        errors.append(np.max(np.abs(theory - numerical)))
    return float(max(errors))


def validate_five_step_propagation(config: RecoverabilityConfig) -> float:
    omega = config.omega
    support = np.zeros(2)
    nominal_b, _ = config.terminal_nominal("left")
    theory_b = nominal_b + np.array((0.018, -0.009))
    numerical_state = state_from_dcm(theory_b, support, omega)
    max_error = 0.0

    for index in range(5):
        if index % 2 == 0:
            cop = np.asarray(config.nominal_cop_left) + np.array((0.008, -0.004))
            landing = np.asarray(config.nominal_step_left) + np.array((0.012, 0.006))
        else:
            cop = np.asarray(config.nominal_cop_right) + np.array((-0.006, 0.003))
            landing = np.asarray(config.nominal_step_right) + np.array((-0.010, -0.005))
        theory_b = propagate_dcm(theory_b, cop, landing, omega, config.step_period)
        numerical_state, support, numerical_b = numerical_touchdown_step(
            numerical_state, support, cop, landing, omega, config.step_period
        )
        max_error = max(max_error, float(np.max(np.abs(theory_b - numerical_b))))
    return max_error


def validate_lp_witness(config: RecoverabilityConfig):
    certificate_state = artificial_states(config)[3]
    result = certify_recoverability(certificate_state, config)
    if result.n_min != 3 or result.witness is None:
        raise AssertionError("The LP witness validation requires the deterministic N_min=3 case.")

    witness = result.witness
    support = np.zeros(2)
    numerical_state = state_from_dcm(certificate_state.b, support, witness.omega)
    q = np.asarray(certificate_state.q).copy()
    max_error = 0.0
    for index in range(3):
        numerical_state, support, numerical_b = numerical_touchdown_step(
            numerical_state,
            support,
            witness.u[index],
            witness.landing[index],
            witness.omega,
            witness.durations[index],
        )
        q = -witness.landing[index]
        max_error = max(max_error, float(np.max(np.abs(numerical_b - witness.b[index + 1]))))

    terminal_ok = terminal_contains(numerical_b, q, witness.support_sides[-1], config)
    if not terminal_ok:
        raise AssertionError("The N_min=3 witness did not enter the terminal set in numerical LIPM.")
    return terminal_ok, max_error, numerical_b, q


def main():
    config = RecoverabilityConfig()
    rng = np.random.default_rng(20260827)

    horizon_results, residual = validate_artificial_horizons(config)
    single_error = validate_single_step(config, rng)
    phase_error = validate_phase(config)
    multistep_error = validate_five_step_propagation(config)
    witness_ok, witness_error, _, _ = validate_lp_witness(config)
    theory_error = max(single_error, phase_error, multistep_error, witness_error)

    print("Artificial recoverability cases:")
    for horizon in (3, 5, 6):
        result = horizon_results[horizon]
        label = ">5" if horizon == 6 else str(horizon)
        print(f"  N_min={label}: PASS, margin={result.margin:+.9f}")
    print(f"LP witness maximum residual: {residual:.3e}")
    print(f"Single-step analytic vs RK4 error: {single_error:.3e}")
    print(f"Phase analytic vs RK4 error: {phase_error:.3e}")
    print(f"Five-step analytic vs RK4 error: {multistep_error:.3e}")
    print(f"LP witness numerical trajectory error: {witness_error:.3e}")
    print(f"Maximum analytic/numerical error: {theory_error:.3e}")
    print(f"LP witness entered terminal set: {'PASS' if witness_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
