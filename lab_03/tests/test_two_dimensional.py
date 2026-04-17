import numpy as np

from two_dimensional import fit_2d_polynomial, predict_2d_polynomial


def test_fit_2d_polynomial_degree2_exact_surface():
    x = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 1.5])

    z = 1.0 + 2.0 * x - 3.0 * y + 0.5 * x * y + 0.2 * x**2 - 0.4 * y**2
    weights = np.ones_like(x)

    fit_result = fit_2d_polynomial(x, y, z, weights, degree=2)
    predicted = predict_2d_polynomial(x, y, fit_result)

    assert np.allclose(predicted, z, atol=1e-9)
