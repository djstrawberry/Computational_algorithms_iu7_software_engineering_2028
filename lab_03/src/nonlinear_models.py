import numpy as np

from lsm_core import solve_weighted_least_squares, weighted_mse


def fit_power_model(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict:
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Для модели a*x^b нужны x > 0 и y > 0.")

    transformed_x = np.column_stack([np.ones_like(x), np.log(x)])
    transformed_y = np.log(y)

    beta = solve_weighted_least_squares(transformed_x, transformed_y, weights)
    a = float(np.exp(beta[0]))
    b = float(beta[1])

    predicted = a * (x ** b)

    return {
        "name": "a*x^b",
        "params": {"a": a, "b": b},
        "predicted": predicted,
        "mse": weighted_mse(y, predicted, weights),
    }


def fit_exponential_model(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict:
    if np.any(y <= 0):
        raise ValueError("Для модели a*exp(b*x) нужны y > 0.")

    transformed_x = np.column_stack([np.ones_like(x), x])
    transformed_y = np.log(y)

    beta = solve_weighted_least_squares(transformed_x, transformed_y, weights)
    a = float(np.exp(beta[0]))
    b = float(beta[1])

    predicted = a * np.exp(b * x)

    return {
        "name": "a*exp(b*x)",
        "params": {"a": a, "b": b},
        "predicted": predicted,
        "mse": weighted_mse(y, predicted, weights),
    }


def fit_inverse_model(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict:
    if np.any(x == 0):
        raise ValueError("Для модели a + b/x нужны x != 0.")

    transformed_x = np.column_stack([np.ones_like(x), 1.0 / x])
    beta = solve_weighted_least_squares(transformed_x, y, weights)

    a = float(beta[0])
    b = float(beta[1])

    predicted = a + b / x

    return {
        "name": "a + b/x",
        "params": {"a": a, "b": b},
        "predicted": predicted,
        "mse": weighted_mse(y, predicted, weights),
    }


def fit_rational_model(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict:
    if np.any(x == 0) or np.any(y == 0):
        raise ValueError("Для модели x/(p + q*x) нужны x != 0 и y != 0.")

    transformed_x = np.column_stack([1.0 / x, np.ones_like(x)])
    transformed_y = 1.0 / y

    beta = solve_weighted_least_squares(transformed_x, transformed_y, weights)

    p = float(beta[0])
    q = float(beta[1])

    denominator = p + q * x
    if np.any(np.abs(denominator) < 1e-14):
        raise ValueError("Знаменатель модели x/(p + q*x) слишком близок к нулю.")

    predicted = x / denominator

    return {
        "name": "x/(p + q*x)",
        "params": {"p": p, "q": q},
        "predicted": predicted,
        "mse": weighted_mse(y, predicted, weights),
    }


def fit_all_models(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> list[dict]:
    fitters = [
        fit_power_model,
        fit_exponential_model,
        fit_inverse_model,
        fit_rational_model,
    ]

    results = []
    for fitter in fitters:
        try:
            results.append(fitter(x, y, weights))
        except ValueError as error:
            results.append(
                {
                    "name": fitter.__name__,
                    "params": {},
                    "predicted": np.full_like(y, np.nan, dtype=float),
                    "mse": float("inf"),
                    "error": str(error),
                }
            )

    return sorted(results, key=lambda item: item["mse"])
