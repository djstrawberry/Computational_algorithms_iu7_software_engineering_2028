"""Численное интегрирование."""

import numpy as np
from typing import Callable


def simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 1000
) -> float:
    """
    Метод Симпсона для численного интегрирования.
    n должно быть чётным.
    """
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    result = y[0] + y[-1]
    result += 4.0 * np.sum(y[1:-1:2])
    result += 2.0 * np.sum(y[2:-2:2])
    result *= h / 3.0

    return result


def trapezoidal(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 1000
) -> float:
    """Метод трапеций."""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    result = (y[0] + y[-1]) / 2.0 + np.sum(y[1:-1])
    result *= h

    return result


def midpoint(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 1000
) -> float:
    """Метод средних прямоугольников."""
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    y = f(x)

    return h * np.sum(y)
