"""Small 1--5 touchdown LIPM/DCM recoverability certificate.

This module is intentionally independent of Isaac Lab.  It only operates on
two-dimensional states expressed in one frozen heading frame and solves small
linear programs with SciPy/HiGHS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from scipy.optimize import OptimizeResult, linprog

SupportSide = Literal["left", "right"]


@dataclass(frozen=True)
class HalfspaceRegion2D:
    """A normalized two-dimensional polytope ``normals @ x <= offsets``."""

    normals: np.ndarray
    offsets: np.ndarray
    scales: np.ndarray

    def __post_init__(self):
        normals = np.asarray(self.normals, dtype=np.float64)
        offsets = np.asarray(self.offsets, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        if normals.ndim != 2 or normals.shape[1] != 2:
            raise ValueError("Region normals must have shape [num_faces, 2].")
        if offsets.shape != (normals.shape[0],) or scales.shape != offsets.shape:
            raise ValueError("Region offsets and scales must match the number of faces.")
        row_norms = np.linalg.norm(normals, axis=1)
        if np.any(row_norms <= 0.0) or np.any(scales <= 0.0):
            raise ValueError("Region normals and normalization scales must be positive.")
        if not np.all(np.isfinite(normals)) or not np.all(np.isfinite(offsets)):
            raise ValueError("Region coefficients must be finite.")

        # Unit normals make offsets and scales physical distances in meters.
        object.__setattr__(self, "normals", normals / row_norms[:, None])
        object.__setattr__(self, "offsets", offsets / row_norms)
        object.__setattr__(self, "scales", scales / row_norms)

    @classmethod
    def box(cls, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> "HalfspaceRegion2D":
        """Construct an axis-aligned box with paired-face normalization scales."""
        x_lo, x_hi = x_bounds
        y_lo, y_hi = y_bounds
        if not x_lo < x_hi or not y_lo < y_hi:
            raise ValueError("Box lower bounds must be strictly smaller than upper bounds.")
        x_scale = 0.5 * (x_hi - x_lo)
        y_scale = 0.5 * (y_hi - y_lo)
        return cls(
            normals=np.array(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))),
            offsets=np.array((x_hi, -x_lo, y_hi, -y_lo)),
            scales=np.array((x_scale, x_scale, y_scale, y_scale)),
        )


@dataclass
class RecoverabilityConfig:
    """All model and capability parameters used by the certificate.

    Every numerical value below is a TEMPORARY DEFAULT / TO BE CALIBRATED.
    The regions are only placeholders that make mathematical validation and the
    first G1 privileged-state data-path test executable.  They are not claimed
    to be physical G1 capability limits.
    """

    gravity: float = 9.81
    h_eff: float = 0.78
    step_period: float = 0.60
    max_steps: int = 5

    cop_left: HalfspaceRegion2D = field(
        default_factory=lambda: HalfspaceRegion2D.box((-0.08, 0.12), (-0.04, 0.04))
    )
    cop_right: HalfspaceRegion2D = field(
        default_factory=lambda: HalfspaceRegion2D.box((-0.08, 0.12), (-0.04, 0.04))
    )
    # A left support places the next right foot at negative lateral y.
    landing_left: HalfspaceRegion2D = field(
        default_factory=lambda: HalfspaceRegion2D.box((-0.25, 0.55), (-0.35, -0.12))
    )
    landing_right: HalfspaceRegion2D = field(
        default_factory=lambda: HalfspaceRegion2D.box((-0.25, 0.55), (0.12, 0.35))
    )
    swing_velocity_limits: tuple[float, float] = (1.20, 0.80)

    nominal_cop_left: tuple[float, float] = (0.0, 0.0)
    nominal_cop_right: tuple[float, float] = (0.0, 0.0)
    nominal_step_left: tuple[float, float] = (0.25, -0.22)
    nominal_step_right: tuple[float, float] = (0.25, 0.22)
    nominal_b_left: tuple[float, float] | None = None
    nominal_b_right: tuple[float, float] | None = None
    nominal_q_left: tuple[float, float] | None = None
    nominal_q_right: tuple[float, float] | None = None
    epsilon_b: tuple[float, float] = (0.05, 0.04)
    epsilon_q: tuple[float, float] = (0.08, 0.05)

    rho_max: float = 0.95
    eta_max: float = 3.0
    feasibility_tolerance: float = 1.0e-7

    def __post_init__(self):
        if self.gravity <= 0.0 or self.h_eff <= 0.0 or self.step_period <= 0.0:
            raise ValueError("gravity, h_eff, and step_period must be positive.")
        if not 1 <= self.max_steps <= 5:
            raise ValueError("max_steps must be between 1 and 5.")
        if any(value <= 0.0 for value in (*self.swing_velocity_limits, *self.epsilon_b, *self.epsilon_q)):
            raise ValueError("Swing limits and terminal tolerances must be positive.")
        if not 0.0 < self.rho_max < 1.0 or self.eta_max <= 0.0:
            raise ValueError("rho_max must be in (0, 1), and eta_max must be positive.")

        # Derive a consistent temporary periodic walking terminal when no
        # explicit nominal b/q values have been calibrated yet.
        gain = math.exp(self.omega * self.step_period)
        u_left = np.asarray(self.nominal_cop_left, dtype=np.float64)
        u_right = np.asarray(self.nominal_cop_right, dtype=np.float64)
        l_left = np.asarray(self.nominal_step_left, dtype=np.float64)
        l_right = np.asarray(self.nominal_step_right, dtype=np.float64)
        d_left = (gain - 1.0) * u_left + l_left
        d_right = (gain - 1.0) * u_right + l_right
        denominator = gain * gain - 1.0
        if self.nominal_b_left is None:
            self.nominal_b_left = tuple((gain * d_left + d_right) / denominator)
        if self.nominal_b_right is None:
            self.nominal_b_right = tuple((d_left + gain * d_right) / denominator)
        if self.nominal_q_left is None:
            self.nominal_q_left = tuple(-l_right)
        if self.nominal_q_right is None:
            self.nominal_q_right = tuple(-l_left)

    @property
    def omega(self) -> float:
        return math.sqrt(self.gravity / self.h_eff)

    def cop_region(self, side: SupportSide) -> HalfspaceRegion2D:
        return self.cop_left if side == "left" else self.cop_right

    def landing_region(self, side: SupportSide) -> HalfspaceRegion2D:
        return self.landing_left if side == "left" else self.landing_right

    def terminal_nominal(self, side: SupportSide) -> tuple[np.ndarray, np.ndarray]:
        if side == "left":
            return np.asarray(self.nominal_b_left), np.asarray(self.nominal_q_left)
        return np.asarray(self.nominal_b_right), np.asarray(self.nominal_q_right)


@dataclass(frozen=True)
class CertificateState:
    """Measured extended DCM state in a frozen heading frame."""

    b: np.ndarray
    q: np.ndarray
    support_side: SupportSide
    phase: float = 0.0
    step_period: float | None = None
    omega: float | None = None


class CertificateStatus(str, Enum):
    FINITE = "finite"
    OVER_HORIZON = "over_horizon"
    INVALID_INPUT = "invalid_input"
    SOLVER_FAILURE = "solver_failure"
    CONSTRAINT_BUILDER_MISMATCH = "constraint_builder_mismatch"


@dataclass(frozen=True)
class CertificateWitness:
    """One LP solution retained only for debugging and residual checks."""

    b: np.ndarray
    q: np.ndarray
    u: np.ndarray
    landing: np.ndarray
    durations: np.ndarray
    support_sides: tuple[SupportSide, ...]
    omega: float
    step_period: float
    phase: float
    kind: Literal["feasible", "inset", "relaxed"]


@dataclass(frozen=True)
class WitnessResidual:
    max_equality: float
    max_inequality_violation: float
    max_dynamics: float
    max_support_switch: float


@dataclass(frozen=True)
class MarginZeroResidual:
    """Residual of a feasibility witness in the inset LP at ``rho=0``."""

    max_equality: float
    max_inequality_violation: float
    max_bound_violation: float
    tolerance: float
    satisfied: bool


@dataclass(frozen=True)
class CertificateResult:
    """Recoverability result; ``n_min == 6`` represents ``>5``."""

    status: CertificateStatus
    n_min: int | None
    margin: float | None
    witness: CertificateWitness | None
    feasible_horizons: tuple[bool, ...]
    message: str = ""
    margin_saturated: bool = False
    margin_fallback: bool = False
    solver_fallback: bool = False
    solver_retried: bool = False
    diagnostic: dict[str, object] | None = None


def opposite_side(side: SupportSide) -> SupportSide:
    return "right" if side == "left" else "left"


def propagate_dcm(
    b: np.ndarray, u: np.ndarray, landing: np.ndarray, omega: float, duration: float
) -> np.ndarray:
    """Apply one exact constant-CoP DCM touchdown map."""
    gain = math.exp(omega * duration)
    return gain * np.asarray(b) - (gain - 1.0) * np.asarray(u) - np.asarray(landing)


class _VariableLayout:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.b_offset = 0
        self.q_offset = 2 * (horizon + 1)
        self.u_offset = self.q_offset + 2 * (horizon + 1)
        self.l_offset = self.u_offset + 2 * horizon
        self.size = self.l_offset + 2 * horizon

    def b(self, index: int) -> slice:
        return slice(self.b_offset + 2 * index, self.b_offset + 2 * index + 2)

    def q(self, index: int) -> slice:
        return slice(self.q_offset + 2 * index, self.q_offset + 2 * index + 2)

    def u(self, index: int) -> slice:
        return slice(self.u_offset + 2 * index, self.u_offset + 2 * index + 2)

    def landing(self, index: int) -> slice:
        return slice(self.l_offset + 2 * index, self.l_offset + 2 * index + 2)


@dataclass(frozen=True)
class _LinearProblem:
    layout: _VariableLayout
    a_ub: np.ndarray
    b_ub: np.ndarray
    scales: np.ndarray
    a_eq: np.ndarray
    b_eq: np.ndarray
    durations: np.ndarray
    support_sides: tuple[SupportSide, ...]
    omega: float
    step_period: float
    phase: float


def _resolved_state(state: CertificateState, config: RecoverabilityConfig):
    b = np.asarray(state.b, dtype=np.float64)
    q = np.asarray(state.q, dtype=np.float64)
    period = config.step_period if state.step_period is None else float(state.step_period)
    omega = config.omega if state.omega is None else float(state.omega)
    if b.shape != (2,) or q.shape != (2,):
        raise ValueError("b and q must each have shape (2,).")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(q)):
        raise ValueError("b and q must be finite.")
    if state.support_side not in ("left", "right"):
        raise ValueError("support_side must be 'left' or 'right'.")
    if not 0.0 <= state.phase < 1.0:
        raise ValueError("phase must lie in [0, 1).")
    if not math.isfinite(period) or not math.isfinite(omega) or period <= 0.0 or omega <= 0.0:
        raise ValueError("step_period and omega must be finite and positive.")
    return b, q, period, omega


def _terminal_rows(
    layout: _VariableLayout,
    index: int,
    side: SupportSide,
    config: RecoverabilityConfig,
) -> tuple[list[np.ndarray], list[float], list[float]]:
    nominal_b, nominal_q = config.terminal_nominal(side)
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    scales: list[float] = []
    for variable_slice, nominal, epsilon in (
        (layout.b(index), nominal_b, np.asarray(config.epsilon_b)),
        (layout.q(index), nominal_q, np.asarray(config.epsilon_q)),
    ):
        for axis in range(2):
            for sign in (1.0, -1.0):
                row = np.zeros(layout.size)
                row[variable_slice.start + axis] = sign
                rows.append(row)
                bounds.append(float(sign * nominal[axis] + epsilon[axis]))
                scales.append(float(epsilon[axis]))
    return rows, bounds, scales


def _build_problem(state: CertificateState, horizon: int, config: RecoverabilityConfig) -> _LinearProblem:
    if not 1 <= horizon <= config.max_steps:
        raise ValueError(f"horizon must be between 1 and {config.max_steps}.")
    b0, q0, period, omega = _resolved_state(state, config)
    durations = np.full(horizon, period, dtype=np.float64)
    durations[0] = (1.0 - state.phase) * period
    sides: list[SupportSide] = [state.support_side]
    for _ in range(horizon):
        sides.append(opposite_side(sides[-1]))

    layout = _VariableLayout(horizon)
    eq_rows: list[np.ndarray] = []
    eq_bounds: list[float] = []
    for variable_slice, value in ((layout.b(0), b0), (layout.q(0), q0)):
        for axis in range(2):
            row = np.zeros(layout.size)
            row[variable_slice.start + axis] = 1.0
            eq_rows.append(row)
            eq_bounds.append(float(value[axis]))

    ub_rows: list[np.ndarray] = []
    ub_bounds: list[float] = []
    ub_scales: list[float] = []

    def add_region(variable_slice: slice, region: HalfspaceRegion2D):
        for normal, offset, scale in zip(region.normals, region.offsets, region.scales):
            row = np.zeros(layout.size)
            row[variable_slice] = normal
            ub_rows.append(row)
            ub_bounds.append(float(offset))
            ub_scales.append(float(scale))

    for index in range(horizon):
        side = sides[index]
        gain = math.exp(omega * durations[index])
        for axis in range(2):
            dynamics = np.zeros(layout.size)
            dynamics[layout.b(index + 1).start + axis] = 1.0
            dynamics[layout.b(index).start + axis] = -gain
            dynamics[layout.u(index).start + axis] = gain - 1.0
            dynamics[layout.landing(index).start + axis] = 1.0
            eq_rows.append(dynamics)
            eq_bounds.append(0.0)

            switch = np.zeros(layout.size)
            switch[layout.q(index + 1).start + axis] = 1.0
            switch[layout.landing(index).start + axis] = 1.0
            eq_rows.append(switch)
            eq_bounds.append(0.0)

        add_region(layout.u(index), config.cop_region(side))
        add_region(layout.landing(index), config.landing_region(side))

        swing_bound = np.asarray(config.swing_velocity_limits) * durations[index]
        for axis in range(2):
            for sign in (1.0, -1.0):
                row = np.zeros(layout.size)
                row[layout.landing(index).start + axis] = sign
                row[layout.q(index).start + axis] = -sign
                ub_rows.append(row)
                ub_bounds.append(float(swing_bound[axis]))
                # The same physical scale is used for the two signs of an axis.
                ub_scales.append(float(max(swing_bound[axis], 1.0e-9)))

    terminal_rows, terminal_bounds, terminal_scales = _terminal_rows(
        layout, horizon, sides[horizon], config
    )
    ub_rows.extend(terminal_rows)
    ub_bounds.extend(terminal_bounds)
    ub_scales.extend(terminal_scales)

    return _LinearProblem(
        layout=layout,
        a_ub=np.asarray(ub_rows),
        b_ub=np.asarray(ub_bounds),
        scales=np.asarray(ub_scales),
        a_eq=np.asarray(eq_rows),
        b_eq=np.asarray(eq_bounds),
        durations=durations,
        support_sides=tuple(sides),
        omega=omega,
        step_period=period,
        phase=state.phase,
    )


def _solve_feasibility(problem: _LinearProblem, *, retry: bool = False) -> OptimizeResult:
    variable_count = problem.layout.size
    return linprog(
        np.zeros(variable_count),
        A_ub=problem.a_ub,
        b_ub=problem.b_ub,
        A_eq=problem.a_eq,
        b_eq=problem.b_eq,
        bounds=[(None, None)] * variable_count,
        method="highs",
        options={"presolve": False} if retry else None,
    )


def _margin_problem_matrices(
    problem: _LinearProblem,
    mode: Literal["inset", "relaxed"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sign = 1.0 if mode == "inset" else -1.0
    return (
        np.column_stack((problem.a_ub, sign * problem.scales)),
        problem.b_ub,
        np.column_stack((problem.a_eq, np.zeros(problem.a_eq.shape[0]))),
        problem.b_eq,
    )


def _solve_with_margin_bound(
    problem: _LinearProblem,
    mode: Literal["inset", "relaxed"],
    bound: float,
    *,
    retry: bool = False,
) -> OptimizeResult:
    variable_count = problem.layout.size
    objective_sign = -1.0 if mode == "inset" else 1.0
    a_ub, b_ub, a_eq, b_eq = _margin_problem_matrices(problem, mode)
    return linprog(
        np.r_[np.zeros(variable_count), objective_sign],
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(None, None)] * variable_count + [(0.0, bound)],
        method="highs",
        options={"presolve": False} if retry else None,
    )


def _solver_summary(result: OptimizeResult) -> dict[str, object]:
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nit": int(getattr(result, "nit", 0) or 0),
    }


def _margin_zero_residual(
    problem: _LinearProblem,
    feasibility_solution: OptimizeResult,
    tolerance: float,
    margin_bound: float,
) -> MarginZeroResidual:
    """Check the exact inset-LP matrices with the feasibility witness and rho=0."""

    values = np.r_[feasibility_solution.x[: problem.layout.size], 0.0]
    a_ub, b_ub, a_eq, b_eq = _margin_problem_matrices(problem, "inset")
    equality = a_eq @ values - b_eq
    inequality = a_ub @ values - b_ub
    rho = float(values[-1])
    max_equality = float(np.max(np.abs(equality), initial=0.0))
    max_inequality = float(max(0.0, np.max(inequality, initial=0.0)))
    max_bound = float(max(0.0, -rho, rho - margin_bound))
    return MarginZeroResidual(
        max_equality=max_equality,
        max_inequality_violation=max_inequality,
        max_bound_violation=max_bound,
        tolerance=float(tolerance),
        satisfied=max(max_equality, max_inequality, max_bound) <= tolerance,
    )


def _witness_to_dict(witness: CertificateWitness | None) -> dict[str, object] | None:
    if witness is None:
        return None
    return {
        "b": witness.b.tolist(),
        "q": witness.q.tolist(),
        "u": witness.u.tolist(),
        "landing": witness.landing.tolist(),
        "durations": witness.durations.tolist(),
        "support_sides": list(witness.support_sides),
        "omega": float(witness.omega),
        "step_period": float(witness.step_period),
        "phase": float(witness.phase),
        "kind": witness.kind,
    }


def _residual_to_dict(residual: WitnessResidual) -> dict[str, float]:
    return {
        "max_equality": residual.max_equality,
        "max_inequality_violation": residual.max_inequality_violation,
        "max_dynamics": residual.max_dynamics,
        "max_support_switch": residual.max_support_switch,
    }


def _margin_zero_residual_to_dict(residual: MarginZeroResidual) -> dict[str, object]:
    return {
        "max_equality": residual.max_equality,
        "max_inequality_violation": residual.max_inequality_violation,
        "max_bound_violation": residual.max_bound_violation,
        "tolerance": residual.tolerance,
        "satisfied": residual.satisfied,
    }


def _witness(problem: _LinearProblem, solution: OptimizeResult, kind: str) -> CertificateWitness:
    values = solution.x[: problem.layout.size]
    horizon = problem.layout.horizon
    return CertificateWitness(
        b=np.stack([values[problem.layout.b(index)] for index in range(horizon + 1)]),
        q=np.stack([values[problem.layout.q(index)] for index in range(horizon + 1)]),
        u=np.stack([values[problem.layout.u(index)] for index in range(horizon)]),
        landing=np.stack([values[problem.layout.landing(index)] for index in range(horizon)]),
        durations=problem.durations.copy(),
        support_sides=problem.support_sides,
        omega=problem.omega,
        step_period=problem.step_period,
        phase=problem.phase,
        kind=kind,  # type: ignore[arg-type]
    )


def _witness_residual(
    problem: _LinearProblem,
    witness: CertificateWitness,
) -> WitnessResidual:
    values = np.zeros(problem.layout.size)
    horizon = problem.layout.horizon
    for index in range(horizon + 1):
        values[problem.layout.b(index)] = witness.b[index]
        values[problem.layout.q(index)] = witness.q[index]
    for index in range(horizon):
        values[problem.layout.u(index)] = witness.u[index]
        values[problem.layout.landing(index)] = witness.landing[index]

    equality = problem.a_eq @ values - problem.b_eq
    inequality = problem.a_ub @ values - problem.b_ub
    dynamics_residuals = []
    switch_residuals = []
    for index in range(horizon):
        propagated = propagate_dcm(
            witness.b[index],
            witness.u[index],
            witness.landing[index],
            witness.omega,
            witness.durations[index],
        )
        dynamics_residuals.append(np.max(np.abs(witness.b[index + 1] - propagated)))
        switch_residuals.append(np.max(np.abs(witness.q[index + 1] + witness.landing[index])))
    return WitnessResidual(
        max_equality=float(np.max(np.abs(equality))),
        max_inequality_violation=float(max(0.0, np.max(inequality))),
        max_dynamics=float(max(dynamics_residuals, default=0.0)),
        max_support_switch=float(max(switch_residuals, default=0.0)),
    )


def _terminal_membership(
    b: np.ndarray, q: np.ndarray, side: SupportSide, config: RecoverabilityConfig
) -> tuple[bool, float]:
    nominal_b, nominal_q = config.terminal_nominal(side)
    normalized_slack = np.r_[
        1.0 - np.abs(b - nominal_b) / np.asarray(config.epsilon_b),
        1.0 - np.abs(q - nominal_q) / np.asarray(config.epsilon_q),
    ]
    feasible = bool(np.min(normalized_slack) >= -config.feasibility_tolerance)
    margin = float(np.clip(np.min(normalized_slack), 0.0, config.rho_max)) if feasible else 0.0
    return feasible, margin


def certify_recoverability(
    state: CertificateState, config: RecoverabilityConfig | None = None
) -> CertificateResult:
    """Return the first feasible touchdown horizon and its signed margin."""
    config = config or RecoverabilityConfig()
    try:
        b0, q0, _, _ = _resolved_state(state, config)
    except (TypeError, ValueError) as exc:
        return CertificateResult(CertificateStatus.INVALID_INPUT, None, None, None, (), str(exc))

    terminal_now, terminal_margin = _terminal_membership(b0, q0, state.support_side, config)
    if terminal_now:
        return CertificateResult(
            CertificateStatus.FINITE, 0, terminal_margin, None, (True,), "Current state is terminal."
        )

    feasibility: list[bool] = [False]
    last_problem: _LinearProblem | None = None
    solver_retried = False
    for horizon in range(1, config.max_steps + 1):
        problem = _build_problem(state, horizon, config)
        last_problem = problem
        feasible = _solve_feasibility(problem)
        if feasible.status == 2:  # HiGHS: reliably infeasible
            feasibility.append(False)
            continue
        if not feasible.success:
            solver_retried = True
            feasible_retry = _solve_feasibility(problem, retry=True)
            if feasible_retry.status == 2:
                feasibility.append(False)
                continue
            if not feasible_retry.success:
                diagnostic = {
                    "kind": "feasibility_solver_failure",
                    "attempted_horizon": horizon,
                    "solver": {
                        "initial": _solver_summary(feasible),
                        "retry": _solver_summary(feasible_retry),
                    },
                    "witness": None,
                    "witness_residual": None,
                    "margin_zero_residual": None,
                }
                return CertificateResult(
                    CertificateStatus.OVER_HORIZON,
                    6,
                    -config.eta_max,
                    None,
                    tuple(feasibility),
                    "Feasibility solver failed twice; conservative fallback.",
                    margin_saturated=True,
                    solver_fallback=True,
                    solver_retried=True,
                    diagnostic=diagnostic,
                )
            feasible = feasible_retry

        feasibility.append(True)
        inset = _solve_with_margin_bound(problem, "inset", config.rho_max)
        if not inset.success:
            feasibility_witness = _witness(problem, feasible, "feasible")
            witness_residual = _witness_residual(problem, feasibility_witness)
            margin_zero_residual = _margin_zero_residual(
                problem,
                feasible,
                config.feasibility_tolerance,
                config.rho_max,
            )
            diagnostic = {
                "kind": "margin_solver_inconsistency",
                "attempted_horizon": horizon,
                "solver": {"initial": _solver_summary(inset)},
                "witness": _witness_to_dict(feasibility_witness),
                "witness_residual": _residual_to_dict(witness_residual),
                "margin_zero_residual": _margin_zero_residual_to_dict(margin_zero_residual),
            }
            if not margin_zero_residual.satisfied:
                return CertificateResult(
                    CertificateStatus.CONSTRAINT_BUILDER_MISMATCH,
                    None,
                    None,
                    feasibility_witness,
                    tuple(feasibility),
                    "Feasibility witness violates inset-LP constraints at rho=0.",
                    solver_retried=solver_retried,
                    diagnostic=diagnostic,
                )

            solver_retried = True
            inset_retry = _solve_with_margin_bound(
                problem,
                "inset",
                config.rho_max,
                retry=True,
            )
            diagnostic["solver"]["retry"] = _solver_summary(inset_retry)
            if not inset_retry.success:
                return CertificateResult(
                    CertificateStatus.FINITE,
                    horizon,
                    0.0,
                    feasibility_witness,
                    tuple(feasibility),
                    "Inset solver failed twice although the rho=0 witness is valid; margin fallback.",
                    margin_fallback=True,
                    solver_retried=True,
                    diagnostic=diagnostic,
                )
            inset = inset_retry
        return CertificateResult(
            CertificateStatus.FINITE,
            horizon,
            float(max(0.0, inset.x[-1])),
            _witness(problem, inset, "inset"),
            tuple(feasibility),
            solver_retried=solver_retried,
        )

    assert last_problem is not None
    relaxed = _solve_with_margin_bound(last_problem, "relaxed", config.eta_max)
    if relaxed.status == 2:
        # The required normalized relaxation lies beyond the configured finite
        # search cap.  Preserve OVER_HORIZON and report a saturated margin.
        return CertificateResult(
            CertificateStatus.OVER_HORIZON,
            6,
            -config.eta_max,
            None,
            tuple(feasibility),
            "Relaxation exceeded eta_max.",
            margin_saturated=True,
            solver_retried=solver_retried,
        )
    if not relaxed.success:
        solver_retried = True
        relaxed_retry = _solve_with_margin_bound(
            last_problem,
            "relaxed",
            config.eta_max,
            retry=True,
        )
        if relaxed_retry.status == 2:
            return CertificateResult(
                CertificateStatus.OVER_HORIZON,
                6,
                -config.eta_max,
                None,
                tuple(feasibility),
                "Relaxation exceeded eta_max on retry.",
                margin_saturated=True,
                solver_retried=True,
            )
        if not relaxed_retry.success:
            diagnostic = {
                "kind": "relaxed_margin_solver_failure",
                "attempted_horizon": config.max_steps,
                "solver": {
                    "initial": _solver_summary(relaxed),
                    "retry": _solver_summary(relaxed_retry),
                },
                "witness": None,
                "witness_residual": None,
                "margin_zero_residual": None,
            }
            return CertificateResult(
                CertificateStatus.OVER_HORIZON,
                6,
                -config.eta_max,
                None,
                tuple(feasibility),
                "Relaxed-margin solver failed twice; conservative fallback.",
                margin_saturated=True,
                solver_fallback=True,
                solver_retried=True,
                diagnostic=diagnostic,
            )
        relaxed = relaxed_retry
    return CertificateResult(
        CertificateStatus.OVER_HORIZON,
        6,
        -float(max(0.0, relaxed.x[-1])),
        _witness(last_problem, relaxed, "relaxed"),
        tuple(feasibility),
        solver_retried=solver_retried,
    )


def check_witness(
    state: CertificateState,
    witness: CertificateWitness,
    config: RecoverabilityConfig | None = None,
) -> WitnessResidual:
    """Rebuild the LP and report independent equality/inequality residuals."""
    config = config or RecoverabilityConfig()
    horizon = witness.u.shape[0]
    problem = _build_problem(state, horizon, config)
    return _witness_residual(problem, witness)


def terminal_contains(
    b: np.ndarray,
    q: np.ndarray,
    support_side: SupportSide,
    config: RecoverabilityConfig | None = None,
) -> bool:
    """Public terminal-set membership check used by independent validation."""
    config = config or RecoverabilityConfig()
    return _terminal_membership(np.asarray(b), np.asarray(q), support_side, config)[0]


__all__ = [
    "CertificateResult",
    "CertificateState",
    "CertificateStatus",
    "CertificateWitness",
    "HalfspaceRegion2D",
    "MarginZeroResidual",
    "RecoverabilityConfig",
    "WitnessResidual",
    "certify_recoverability",
    "check_witness",
    "opposite_side",
    "propagate_dcm",
    "terminal_contains",
]
