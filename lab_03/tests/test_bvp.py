import numpy as np

from bvp import solve_bvp_by_least_squares


def test_bvp_solution_satisfies_boundary_conditions():
    solution = solve_bvp_by_least_squares(m=3, grid_points=301)

    y = solution["y"]

    assert abs(y[0] - 1.0) < 1e-10
    assert abs(y[-1] - 0.0) < 1e-10


def test_bvp_residual_improves_with_more_basis():
    solution_m2 = solve_bvp_by_least_squares(m=2, grid_points=301)
    solution_m3 = solve_bvp_by_least_squares(m=3, grid_points=301)

    assert np.isfinite(solution_m2["residual_l2"])
    assert np.isfinite(solution_m3["residual_l2"])
    assert solution_m3["residual_l2"] <= solution_m2["residual_l2"]
