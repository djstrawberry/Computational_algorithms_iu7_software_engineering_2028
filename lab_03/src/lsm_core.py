import numpy as np


def build_1d_polynomial_design(x: np.ndarray, degree: int) -> np.ndarray:
    if degree < 0:
        raise ValueError("Степень полинома должна быть неотрицательной.")

    columns = [x ** power for power in range(degree + 1)]

    return np.column_stack(columns)


def build_2d_polynomial_design(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if degree < 0:
        raise ValueError("Степень полинома должна быть неотрицательной.")

    powers = []
    for total_power in range(degree + 1):
        for x_power in range(total_power, -1, -1):
            y_power = total_power - x_power
            powers.append((x_power, y_power))

    columns = [(x ** x_power) * (y ** y_power) for x_power, y_power in powers]

    return np.column_stack(columns), powers


def solve_weighted_least_squares(design_matrix: np.ndarray, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if design_matrix.shape[0] != values.shape[0] or values.shape[0] != weights.shape[0]:
        raise ValueError("Размеры матрицы признаков, значений и весов должны совпадать по числу строк.")

    if np.any(weights <= 0):
        raise ValueError("Все веса должны быть строго положительными.")

    sqrt_weights = np.sqrt(weights)
    weighted_matrix = design_matrix * sqrt_weights[:, None]
    weighted_values = values * sqrt_weights

    normal_matrix = weighted_matrix.T @ weighted_matrix
    normal_vector = weighted_matrix.T @ weighted_values

    try:
        coefficients = np.linalg.solve(normal_matrix, normal_vector)
    except np.linalg.LinAlgError:
        coefficients, _, _, _ = np.linalg.lstsq(weighted_matrix, weighted_values, rcond=None)

    return coefficients


def evaluate_1d_polynomial(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x, dtype=float)

    for power, coefficient in enumerate(coefficients):
        result += coefficient * (x ** power)

    return result


def evaluate_2d_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    powers: list[tuple[int, int]],
) -> np.ndarray:
    result = np.zeros_like(x, dtype=float)

    for coefficient, (x_power, y_power) in zip(coefficients, powers):
        result += coefficient * (x ** x_power) * (y ** y_power)

    return result


def weighted_mse(real_values: np.ndarray, predicted_values: np.ndarray, weights: np.ndarray) -> float:
    if np.any(weights <= 0):
        raise ValueError("Все веса должны быть строго положительными.")

    squared_error = (real_values - predicted_values) ** 2

    return float(np.sum(weights * squared_error) / np.sum(weights))
