import numpy as np

from lsm_core import (
    build_2d_polynomial_design,
    evaluate_2d_polynomial,
    solve_weighted_least_squares,
)


def base_function_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.2 + 0.9 * x - 0.8 * y + 0.35 * x * y + 0.22 * x**2 - 0.1 * y**2


def generate_2d_table(
    nodes_count: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    noise_sigma: float,
    random_seed: int,
) -> dict:
    random_generator = np.random.default_rng(random_seed)

    x = random_generator.uniform(x_min, x_max, nodes_count)
    y = random_generator.uniform(y_min, y_max, nodes_count)

    z_clean = base_function_2d(x, y)
    noise = random_generator.normal(0.0, noise_sigma, nodes_count)
    z = z_clean + noise

    weights = random_generator.uniform(0.7, 1.4, nodes_count)

    return {
        "x": x,
        "y": y,
        "z": z,
        "weights": weights,
        "z_clean": z_clean,
    }


def fit_2d_polynomial(x: np.ndarray, y: np.ndarray, z: np.ndarray, weights: np.ndarray, degree: int) -> dict:
    design, powers = build_2d_polynomial_design(x, y, degree)
    coefficients = solve_weighted_least_squares(design, z, weights)

    return {
        "degree": degree,
        "coefficients": coefficients,
        "powers": powers,
    }


def predict_2d_polynomial(x: np.ndarray, y: np.ndarray, fit_result: dict) -> np.ndarray:
    return evaluate_2d_polynomial(x, y, fit_result["coefficients"], fit_result["powers"])
