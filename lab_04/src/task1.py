"""Задача 1: Решение системы нелинейных уравнений."""

import numpy as np
from nonlinear_solvers import newton_system


def F1(x: np.ndarray) -> np.ndarray:
    """Система уравнений задачи 1."""
    x_val, y_val = x[0], x[1]
    return np.array([
        20.0 * np.log(x_val - y_val) - x_val - y_val - 6.0,
        20.0 * np.sin(0.7 * x_val - 0.7 * y_val) + 7.0 * x_val + 7.0 * y_val
    ])


def jacobian_F1(x: np.ndarray) -> np.ndarray:
    """Аналитический Якобиан для задачи 1."""
    x_val, y_val = x[0], x[1]
    dx = x_val - y_val

    J = np.zeros((2, 2))
    J[0, 0] = 20.0 / dx - 1.0
    J[0, 1] = -20.0 / dx - 1.0

    cos_arg = 0.7 * x_val - 0.7 * y_val
    J[1, 0] = 20.0 * 0.7 * np.cos(cos_arg) + 7.0
    J[1, 1] = -20.0 * 0.7 * np.cos(cos_arg) + 7.0

    return J


def solve_task1():
    """Решение задачи 1 с разными допусками."""
    print("Задача 1. Решение системы нелинейных уравнений")
    print("-" * 40)
    
    x0 = np.array([3.0, 1.0])
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    results = []

    for tol in tolerances:
        x_sol, history, iters = newton_system(
            F1, x0, tol=tol, max_iter=100,
            use_analytical_jac=jacobian_F1
        )
        results.append({
            'tol': tol,
            'x': x_sol[0],
            'y': x_sol[1],
            'iterations': iters
        })
        print(f"Допуск: {tol:.0e}")
        print(f"x = {x_sol[0]:.10f}, y = {x_sol[1]:.10f}")
        print(f"Итераций: {iters}\n")

    return results
