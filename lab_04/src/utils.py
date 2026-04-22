"""Вспомогательные функции."""

from typing import Callable, Tuple, List
import numpy as np


def finite_difference_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Вычисление матрицы Якоби с помощью конечных разностей.
    """
    n = len(x)
    f0 = func(x)
    m = len(f0)
    J = np.zeros((m, n))
    dx = np.zeros_like(x)

    for j in range(n):
        dx[j] = eps
        f1 = func(x + dx)
        J[:, j] = (f1 - f0) / eps
        dx[j] = 0.0

    return J


def norm(vector: np.ndarray) -> float:
    """Евклидова норма вектора."""
    return float(np.linalg.norm(vector))


def relative_error(x_new: np.ndarray, x_old: np.ndarray) -> float:
    """Относительная ошибка."""
    return norm(x_new - x_old) / (norm(x_new) + 1e-12)
