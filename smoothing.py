"""Minimum-curvature monthly smoothing with exact yearly means and no bounds."""
import cvxpy as cp
import numpy as np
from scipy import sparse


def smooth_average(targets):
    """Solve independent columns using shared sparse operators.

    Scale each column before optimization to support ordinary absolute values
    and very large/small finite inputs without badly scaled solver constraints.
    Neither endpoints nor extrema are clamped or artificially anchored.
    """
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
    problem = cp.Problem(cp.Minimize(cp.sum_squares(differences @ x)),
                         [averages @ x == yearly])
    result = np.empty((months, columns))
    for column in range(columns):
        target = targets[:, column]
        if np.all(target == target[0]):
            result[:, column] = target[0]
            continue
        scale = np.max(np.abs(target))
        normalized = target / scale
        yearly.value = normalized
        x.value = np.repeat(normalized, 12)
        try:
            problem.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10,
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
        with np.errstate(over="ignore", invalid="ignore"):
            result[:, column] = ((blocks + residual[:, None]) * scale).ravel()
    return result
