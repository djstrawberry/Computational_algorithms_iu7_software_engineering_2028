import numpy as np


def poly_for_u0() -> np.poly1d:
    # u0(0)=1 и u0(1)=0
    return np.poly1d([-1.0, 1.0])


def poly_for_uk(k: int) -> np.poly1d:
    if k < 1:
        raise ValueError("Индекс базисной функции должен быть >= 1.")

    # uk(x) = x^k * (1 - x) = x^k - x^(k+1)
    coefficients = np.zeros(k + 2)
    coefficients[0] = -1.0
    coefficients[1] = 1.0

    return np.poly1d(coefficients)


def apply_operator(poly: np.poly1d, x: np.ndarray) -> np.ndarray:
    # По условию используем оператор L[y] = y'' + x*y' + y
    first_derivative = np.polyder(poly, 1)
    second_derivative = np.polyder(poly, 2)

    return second_derivative(x) + x * first_derivative(x) + poly(x)


def solve_bvp_by_least_squares(m: int, grid_points: int = 401) -> dict:
    if m < 1:
        raise ValueError("Число базисных функций m должно быть >= 1.")

    x = np.linspace(0.0, 1.0, grid_points)

    u0 = poly_for_u0()
    right_part = x

    lu0 = apply_operator(u0, x)
    residual_base = right_part - lu0

    basis = [poly_for_uk(k) for k in range(1, m + 1)]
    transformed_basis = [apply_operator(poly, x) for poly in basis]

    normal_matrix = np.zeros((m, m))
    normal_vector = np.zeros(m)

    for i in range(m):
        for j in range(m):
            normal_matrix[i, j] = np.trapezoid(transformed_basis[i] * transformed_basis[j], x)
        normal_vector[i] = np.trapezoid(residual_base * transformed_basis[i], x)

    coefficients = np.linalg.solve(normal_matrix, normal_vector)

    y = u0(x)
    for coefficient, basis_poly in zip(coefficients, basis):
        y += coefficient * basis_poly(x)

    residual = apply_operator(np.poly1d([0.0]), x)
    residual = right_part - apply_operator(u0, x)
    for coefficient, basis_l in zip(coefficients, transformed_basis):
        residual -= coefficient * basis_l

    return {
        "x": x,
        "y": y,
        "coefficients": coefficients,
        "residual_l2": float(np.sqrt(np.trapezoid(residual**2, x))),
    }
