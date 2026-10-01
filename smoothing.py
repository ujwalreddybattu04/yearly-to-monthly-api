"""Minimum-curvature monthly smoothing with exact yearly means and a data-derived soft-range penalty."""
from decimal import Decimal, localcontext
import math

import cvxpy as cp
import numpy as np
from scipy import sparse


# Higher values discourage range excursions; lower values prioritize curvature.
# This weight never imposes a hard min/max bound.
RANGE_PENALTY_WEIGHT = 0.1
FIRST_MONTH_FRACTION = 0.01
FIRST_MONTH_FLOOR = 1e-6


# Leave headroom below float64's roughly 15-17 reliable significant digits.
MAX_SAFE_MAGNITUDE_RATIO = 1e12
MEAN_RELATIVE_TOLERANCE = Decimal("1e-6")
MEAN_ABSOLUTE_FLOOR = Decimal("1e-6")


def _check_magnitude_ratio(target, name):
    nonzero = [abs(float(v)) for v in target if v != 0]
    if len(nonzero) < 2:
        return
    with localcontext() as context:
        context.prec = 800
        ratio = Decimal(str(max(nonzero))) / Decimal(str(min(nonzero)))
        if ratio > Decimal(str(MAX_SAFE_MAGNITUDE_RATIO)):
            raise ValueError(
                f"Column '{name}' has yearly values spanning too wide a range to smooth "
                f"reliably (ratio of largest to smallest value is {ratio:.2e}, limit is "
                f"{MAX_SAFE_MAGNITUDE_RATIO:.0e}). Consider splitting extreme outlier years "
                "into a separate request, or reviewing the input data."
            )


def _verify_monthly_means(values, targets, name):
    """Verify raw returned floats independently, including severe cancellation.

    800 decimal digits cover the entire finite float64 exponent span plus the
    accumulation of twelve values. The floor allows 1e-12 absolute mean error
    for near-zero targets; otherwise the tolerance is relative to each year.
    """
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError(f"Column '{name}' produced a non-finite result; input values are too large.")
    with localcontext() as context:
        context.prec = 800
        for i, block in enumerate(values.reshape(-1, 12)):
            mean = sum((Decimal(str(v)) for v in block), Decimal(0)) / 12
            target = Decimal(str(targets[i]))
            if abs(mean - target) / max(abs(target), MEAN_ABSOLUTE_FLOOR) >= MEAN_RELATIVE_TOLERANCE:
                raise ValueError(
                    f"Column '{name}' failed a post-smoothing accuracy check for year index {i}; "
                    "the result cannot be verified as correct."
                )


def _refine_raw_roundoff(values, targets):
    """Extend equality roundoff cleanup after rescaling; never repair large errors."""
    with localcontext() as context:
        context.prec = 800
        for i, block in enumerate(values.reshape(-1, 12)):
            if not np.isfinite(block).all():
                continue
            decimals = [Decimal(str(v)) for v in block]
            target = Decimal(str(targets[i]))
            error = target * 12 - sum(decimals, Decimal(0))
            tolerance = max(abs(target), MEAN_ABSOLUTE_FLOOR) * MEAN_RELATIVE_TOLERANCE
            if abs(error) / 12 < tolerance:
                continue
            # Restrict changes to numerical noise relative to this year's values.
            if abs(error) > max(map(abs, decimals)) * Decimal("1e-12"):
                continue
            # Preserve the positive first-month constraint exactly.
            candidates = range(1, 12) if i == 0 else range(12)
            index = min(candidates, key=lambda j: abs(decimals[j]))
            block[index] = float(decimals[index] + error)


def smooth_average(targets, column_names=None):
    """Solve independent columns using shared sparse operators.

    Scale each column before optimization to support ordinary absolute values
    and very large/small finite inputs without badly scaled solver constraints.
    Yearly means and positive M1 are hard constraints; input extrema are soft preferences.
    """
    years, columns = targets.shape
    names = list(column_names) if column_names is not None else [f"column_{i}" for i in range(columns)]
    if len(names) != columns:
        raise ValueError("Column names must match the target columns.")
    for column, name in enumerate(names):
        _check_magnitude_ratio(targets[:, column], name)
    epsilons = np.maximum(FIRST_MONTH_FLOOR, FIRST_MONTH_FRACTION * np.maximum(targets.max(axis=0), 0))
    if years == 1 and np.all(targets[0] >= epsilons):
        result = np.repeat(targets, 12, axis=0)
        for column, name in enumerate(names):
            _verify_monthly_means(result[:, column], targets[:, column], name)
        return result
    months = years * 12
    differences = sparse.diags([np.ones(months-2), -2*np.ones(months-2),
                                np.ones(months-2)], [0, 1, 2],
                               shape=(months-2, months), format="csc")
    averages = sparse.kron(sparse.eye(years, format="csc"),
                           sparse.csc_matrix(np.ones((1, 12))/12), format="csc")
    x = cp.Variable(months)
    yearly = cp.Parameter(years)
    first_min = cp.Parameter(nonneg=True)
    data_min, data_max = cp.Parameter(), cp.Parameter()
    below, above = cp.Variable(months, nonneg=True), cp.Variable(months, nonneg=True)
    objective = (cp.sum_squares(differences @ x)
                 + RANGE_PENALTY_WEIGHT * cp.sum_squares(below)
                 + RANGE_PENALTY_WEIGHT * cp.sum_squares(above))
    problem = cp.Problem(cp.Minimize(objective),
                         [averages @ x == yearly, x[0] >= first_min,
                          below >= data_min - x, above >= x - data_max])
    result = np.empty((months, columns))
    for column in range(columns):
        target = targets[:, column]
        epsilon = epsilons[column]
        if np.all(target == target[0]) and target[0] >= epsilon:
            result[:, column] = target[0]
            continue
        scale = max(np.max(np.abs(target)), epsilon) / 100.0
        normalized = target / scale
        yearly.value = normalized
        data_min.value, data_max.value = normalized.min(), normalized.max()
        first_min.value = epsilon / scale
        x.value = np.repeat(normalized, 12)
        try:
            problem.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, rho=0.01, adaptive_rho=False,
                          max_iter=30000, polishing=True, warm_start=True)
        except cp.error.SolverError as exc:
            raise ValueError("Average optimization failed to converge; please retry.") from exc
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
            raise ValueError("Average optimization failed to converge; please retry.")
        blocks = x.value.reshape(years, 12)
        residual = normalized - blocks.mean(axis=1)
        if not np.isfinite(blocks).all() or np.max(np.abs(residual)) > 1e-8:
            raise ValueError("Average optimization did not meet numerical accuracy requirements.")
        # Remove only equality-constraint roundoff in normalized coordinates.
        blocks = blocks + residual[:, None]
        # Repair only inequality roundoff and compensate inside the same year,
        # preserving its mean. Do not clamp any other monthly values.
        correction = max(0.0, first_min.value - blocks[0, 0])
        if correction > 1e-8:
            raise ValueError("Average optimization did not meet first-month accuracy requirements.")
        # Assign directly: adding a tiny epsilon to a cancelling correction can
        # round back to zero at extreme scales.
        blocks[0, 0] = max(blocks[0, 0], first_min.value)
        blocks[0, 1:] -= correction / 11
        with np.errstate(over="ignore", invalid="ignore"):
            result[:, column] = (blocks * scale).ravel()
        if result[0, column] <= 0:
            raise ValueError("Average optimization did not produce a positive first month.")
    for column, name in enumerate(names):
        _refine_raw_roundoff(result[:, column], targets[:, column])
        _verify_monthly_means(result[:, column], targets[:, column], name)
    return result
