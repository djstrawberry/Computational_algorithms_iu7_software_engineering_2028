import numpy as np

from lsm_core import (
    build_1d_polynomial_design,
    build_2d_polynomial_design,
    evaluate_1d_polynomial,
    solve_weighted_least_squares,
)

EPS = 1e-10


def test_weighted_least_squares_1d_exact_line():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 + 3.0 * x
    w = np.ones_like(x)

    design = build_1d_polynomial_design(x, degree=1)
    coeffs = solve_weighted_least_squares(design, y, w)

    assert abs(coeffs[0] - 2.0) < EPS
    assert abs(coeffs[1] - 3.0) < EPS


def test_evaluate_1d_polynomial():
    x = np.array([0.0, 2.0])
    coeffs = np.array([1.0, -2.0, 1.0])

    predicted = evaluate_1d_polynomial(x, coeffs)

    assert abs(predicted[0] - 1.0) < EPS
    assert abs(predicted[1] - 1.0) < EPS


def test_build_2d_design_degree_2_column_count():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])

    design, powers = build_2d_polynomial_design(x, y, degree=2)

    assert design.shape[1] == 6
    assert len(powers) == 6
