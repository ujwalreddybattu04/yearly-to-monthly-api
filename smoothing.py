"""Sparse minimum-curvature smoothing with bounded, exact yearly means."""
import cvxpy as cp
import numpy as np
from scipy import sparse


def _restore_feasibility(values, targets):
    """Remove solver roundoff by projecting each year onto its bounded mean plane.

    This is numerical cleanup, not clipping an unconstrained curve: both the
    bounds and means were enforced in the quadratic program already.
    """
    blocks = values.reshape(-1, 12)
    if (not np.isfinite(blocks).all() or blocks.min() < -1e-5
            or blocks.max() > 100 + 1e-5
            or np.max(np.abs(blocks.mean(axis=1) - targets)) > 1e-5):
        raise ValueError("Average optimization did not meet numerical accuracy requirements.")
    low = np.full(len(targets), -1e-4)
    high = np.full(len(targets), 1e-4)
    for _ in range(50):
        shift = (low + high) / 2
        means = np.clip(blocks + shift[:, None], 0, 100).mean(axis=1)
        low = np.where(means < targets, shift, low)
        high = np.where(means >= targets, shift, high)
    result = np.clip(blocks + ((low + high) / 2)[:, None], 0, 100)
    result[targets == 0] = 0
    result[targets == 100] = 100
    if np.max(np.abs(result.mean(axis=1) - targets)) > 1e-6:
        raise ValueError("Average optimization did not preserve yearly means.")
    return result.ravel()


def bounded_average(targets):
    """Solve each percentage column independently, reusing one sparse QP."""
    years, columns = targets.shape
    if years == 1:
        return np.repeat(targets, 12, axis=0)
    months = years * 12
    differences = sparse.diags([np.ones(months-2), -2*np.ones(months-2),
                                np.ones(months-2)], [0, 1, 2],
                               shape=(months-2, months), format="csc")
    averages = sparse.kron(sparse.eye(years, format="csc"),
                           sparse.csc_matrix(np.ones((1, 12))/12), format="csc")
    x = cp.Variable(months)
    yearly = cp.Parameter(years)
    lower, upper = cp.Parameter(months), cp.Parameter(months)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(differences @ x)),
                         [averages @ x == yearly, x >= lower, x <= upper])
    result = np.empty((months, columns))
    for column in range(columns):
        target = targets[:, column]
        # Constant inputs have the exact zero-curvature solution.
        if np.all(target == target[0]):
            result[:, column] = target[0]
            continue
        yearly.value = target
        lower.value = np.repeat(np.where(target == 100, 100., 0.), 12)
        upper.value = np.repeat(np.where(target == 0, 0., 100.), 12)
        x.value = np.repeat(target, 12)
        try:
            problem.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, adaptive_rho_interval=50,
                          max_iter=30000, polishing=True, warm_start=True)
        except cp.error.SolverError as exc:
            raise ValueError("Average optimization failed to converge; please retry.") from exc
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
            raise ValueError("Average optimization failed to converge; please retry.")
        result[:, column] = _restore_feasibility(x.value, target)
    return result
