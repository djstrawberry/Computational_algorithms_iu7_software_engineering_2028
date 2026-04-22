"""Методы решения нелинейных уравнений и систем."""

from typing import Callable, Tuple, List, Optional
import numpy as np
from utils import finite_difference_jacobian, norm


def newton_system(
    F: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    use_analytical_jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    modified: bool = False
) -> Tuple[np.ndarray, List[np.ndarray], int]:
    """
    Метод Ньютона для систем нелинейных уравнений.

    Параметры:
        F: функция системы
        x0: начальное приближение
        tol: допуск
        max_iter: максимальное число итераций
        use_analytical_jac: аналитический Якобиан (если есть)
        modified: модифицированный метод (без пересчёта Якобиана)

    Возвращает:
        x: решение
        history: история приближений
        iterations: число итераций
    """
    x = x0.copy()
    history = [x.copy()]

    if use_analytical_jac is not None:
        J = use_analytical_jac(x)
    else:
        J = finite_difference_jacobian(F, x)

    for it in range(max_iter):
        Fx = F(x)

        if not modified or it == 0:
            if use_analytical_jac is not None:
                J = use_analytical_jac(x)
            else:
                J = finite_difference_jacobian(F, x)

        try:
            delta = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Если матрица вырождена, используем псевдообратную
            delta = -np.linalg.pinv(J) @ Fx

        x_new = x + delta
        history.append(x_new.copy())

        if norm(delta) < tol:
            return x_new, history, it + 1

        x = x_new

    return x, history, max_iter


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 200
) -> Tuple[float, List[float], int]:
    """
    Метод половинного деления.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Функция должна иметь разные знаки на концах интервала")

    history = [a, b]

    for it in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        history.append(c)

        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c, history, it + 1

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return (a + b) / 2.0, history, max_iter


def secant_method(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, List[float], int]:
    """
    Метод секущих (двухшаговый).
    """
    history = [x0, x1]
    f0 = f(x0)

    for it in range(max_iter):
        f1 = f(x1)

        if abs(f1 - f0) < 1e-12:
            raise ValueError("Деление на ноль в методе секущих")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)

        if abs(x2 - x1) < tol:
            return x2, history, it + 1

        x0, f0 = x1, f1
        x1 = x2

    return x1, history, max_iter


def parabola_method(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    x2: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, List[float], int]:
    """
    Метод парабол (трёхшаговый, Мюллера).
    """
    history = [x0, x1, x2]

    for it in range(max_iter):
        f0, f1, f2 = f(x0), f(x1), f(x2)

        # Построение параболы через три точки
        h1 = x1 - x0
        h2 = x2 - x1
        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        c = f2

        # Дискриминант
        discriminant = b * b - 4.0 * a * c

        if discriminant < 0:
            # Если корни комплексные, берём минимальный по модулю сдвиг
            delta = -c / b if abs(b) > 1e-12 else h2
        else:
            sqrt_disc = np.sqrt(discriminant)
            denom1 = b + sqrt_disc if b > 0 else b - sqrt_disc
            denom2 = b - sqrt_disc if b > 0 else b + sqrt_disc

            dx1 = -2.0 * c / denom1
            dx2 = -2.0 * c / denom2

            # Выбираем наименьший по модулю сдвиг
            delta = dx1 if abs(dx1) < abs(dx2) else dx2

        x3 = x2 + delta
        history.append(x3)

        if abs(delta) < tol:
            return x3, history, it + 1

        # Сохраняем три ближайшие точки
        points = [(x0, abs(f0)), (x1, abs(f1)), (x2, abs(f2)), (x3, abs(f(x3)))]
        points.sort(key=lambda p: p[1])
        x0, x1, x2 = points[0][0], points[1][0], points[2][0]

    return x2, history, max_iter
