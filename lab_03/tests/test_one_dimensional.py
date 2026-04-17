import numpy as np

from one_dimensional import (
    fit_1d_polynomial,
    generate_weight_impact_table,
    parse_weights_line,
    slope_of_line,
)

EPS = 1e-10


def test_parse_weights_line_ok():
    weights = parse_weights_line("1, 2, 3", 3)

    assert np.allclose(weights, np.array([1.0, 2.0, 3.0]))


def test_linear_fit_with_weights_changes_slope():
    table = generate_weight_impact_table()

    fit_uniform = fit_1d_polynomial(table["x"], table["y"], table["uniform_weights"], degree=1)
    fit_custom = fit_1d_polynomial(table["x"], table["y"], table["custom_weights"], degree=1)

    slope_uniform = slope_of_line(fit_uniform)
    slope_custom = slope_of_line(fit_custom)

    assert slope_uniform < 0
    assert slope_custom > 0
    assert abs(slope_uniform - slope_custom) > EPS
