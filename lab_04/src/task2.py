"""Задача 2: Нахождение аргумента функции Лапласа."""

import numpy as np
from integration import simpson
from nonlinear_solvers import bisection


def laplace_integrand(t: float) -> float:
    """Подынтегральная функция для Лапласа."""
    return np.exp(-t**2 / 2.0)


def laplace_function(x: float, n: int = 1000) -> float:
    """Функция Лапласа Φ(x) = 2/√(2π) ∫₀ˣ exp(-t²/2) dt."""
    if x == 0:
        return 0.0
    integral = simpson(laplace_integrand, 0.0, abs(x), n)
    return (2.0 / np.sqrt(2.0 * np.pi)) * integral * np.sign(x)


def solve_task2(target_phi: float = 0.95):
    """Решение задачи 2: найти x, такой что Φ(x) = target_phi."""
    print("\nЗадача 2. Аргумент функции Лапласа")
    print("-" * 40)
    print(f"Φ(x) = {target_phi}")

    def residual(x: float) -> float:
        return laplace_function(x) - target_phi

    a, b = 0.0, 5.0
    fa = residual(a)
    fb = residual(b)

    if fa * fb > 0:
        b = 10.0

    x_sol, history, iters = bisection(residual, a, b, tol=1e-8)
    print(f"x = {x_sol:.10f}\n")

    return x_sol, history, iters
