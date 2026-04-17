import numpy as np

from lsm_core import build_1d_polynomial_design, evaluate_1d_polynomial, solve_weighted_least_squares


def base_function_1d(x: np.ndarray) -> np.ndarray:
    return 0.7 * x**3 - 1.4 * x**2 + 0.5 * x + 1.8


def generate_1d_table(
    nodes_count: int,
    x_min: float,
    x_max: float,
    noise_sigma: float,
    random_seed: int,
) -> dict:
    random_generator = np.random.default_rng(random_seed)

    x = np.sort(random_generator.uniform(x_min, x_max, nodes_count))
    y_clean = base_function_1d(x)
    noise = random_generator.normal(0.0, noise_sigma, nodes_count)
    y = y_clean + noise
    weights = np.ones(nodes_count)

    return {
        "x": x,
        "y": y,
        "weights": weights,
        "y_clean": y_clean,
    }


def create_inverted_weights(base_weights: np.ndarray, first_half_weight: float = 8.0) -> np.ndarray:
    inverted = np.ones_like(base_weights)
    mid = len(base_weights) // 2
    for i in range(len(base_weights)):
        if i < mid:
            inverted[i] = first_half_weight
        else:
            inverted[i] = 1.0 / first_half_weight

    return inverted


def fit_1d_polynomial(x: np.ndarray, y: np.ndarray, weights: np.ndarray, degree: int) -> dict:
    design = build_1d_polynomial_design(x, degree)
    coefficients = solve_weighted_least_squares(design, y, weights)

    return {
        "degree": degree,
        "coefficients": coefficients,
    }


def predict_1d_polynomial(x: np.ndarray, fit_result: dict) -> np.ndarray:
    return evaluate_1d_polynomial(x, fit_result["coefficients"])


def slope_of_line(fit_result: dict) -> float:
    coefficients = fit_result["coefficients"]
    if len(coefficients) < 2:
        return 0.0

    return float(coefficients[1])


def print_weight_table(weights: np.ndarray) -> None:
    print("Текущие веса:")
    for index, weight in enumerate(weights):
        print(f"  [{index}] rho = {weight:.6f}")


def parse_weights_line(user_line: str, expected_count: int) -> np.ndarray:
    pieces = [piece.strip() for piece in user_line.split(",") if piece.strip()]
    if len(pieces) != expected_count:
        raise ValueError("Количество весов должно совпадать с числом точек.")

    weights = np.array([float(piece) for piece in pieces], dtype=float)
    if np.any(weights <= 0):
        raise ValueError("Каждый вес должен быть строго положительным.")

    return weights


def edit_weights_from_cli(default_weights: np.ndarray) -> np.ndarray:
    print("\nРедактирование весов для одномерной таблицы.")
    print("Введите все веса через запятую в одной строке.")
    print("Чтобы оставить веса без изменений, нажмите Enter.")

    print_weight_table(default_weights)

    user_line = input("Новые веса: ").strip()
    if user_line == "":
        return default_weights.copy()

    try:
        return parse_weights_line(user_line, len(default_weights))
    except ValueError as error:
        print(f"Ошибка ввода: {error}")
        print("Возвращаю исходные веса.")
        return default_weights.copy()
