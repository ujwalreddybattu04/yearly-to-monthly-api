"""Minimum-curvature monthly smoothing with exact yearly means and a data-derived soft range."""
import cvxpy as cp
import numpy as np
from scipy import sparse


# Higher weights resist excursions more strongly; lower weights favor curvature.
# This is a soft penalty, not a guarantee of staying within yearly extrema.
RANGE_PENALTY_WEIGHT = 0.1
FIRST_MONTH_FRACTION = 0.01
FIRST_MONTH_FLOOR = 1e-6


def smooth_average(targets):
    """Solve independent columns using shared sparse operators.

    Scale each column before optimization to support ordinary absolute values
    and very large/small finite inputs without badly scaled solver constraints.
    Each column has its own soft min/max and positive first-month constraint.
    """
    years, columns = targets.shape
    epsilons = np.maximum(FIRST_MONTH_FLOOR, FIRST_MONTH_FRACTION * np.maximum(targets.max(axis=0), 0))
    if years == 1 and np.all(targets[0] >= epsilons):
        return np.repeat(targets, 12, axis=0)
    months = years * 12
    differences = sparse.diags([np.ones(months-2), -2*np.ones(months-2),
                                np.ones(months-2)], [0, 1, 2],
                               shape=(months-2, months), format="csc")
    averages = sparse.kron(sparse.eye(years, format="csc"),
                           sparse.csc_matrix(np.ones((1, 12))/12), format="csc")
    x = cp.Variable(months)
    yearly = cp.Parameter(years)
    data_min, data_max, first_min = cp.Parameter(), cp.Parameter(), cp.Parameter(nonneg=True)
    # Explicit nonnegative slacks implement squared hinge loss without the
    # redundant auxiliaries introduced by composing pos() and sum_squares().
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
        blocks[0, 0] += correction
        blocks[0, 1:] -= correction / 11
        with np.errstate(over="ignore", invalid="ignore"):
            result[:, column] = (blocks * scale).ravel()
        if result[0, column] <= 0:
            raise ValueError("Average optimization did not produce a positive first month.")
    return result
