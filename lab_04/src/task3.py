"""Задача 3: Краевая задача для ОДУ методом конечных разностей."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional


def solve_tridiagonal(
    lower: List[float], 
    diag: List[float], 
    upper: List[float], 
    rhs: List[float]
) -> List[float]:
    """Метод прогонки (алгоритм Томаса) для трёхдиагональной системы."""
    n = len(diag)
    if n == 0:
        return []
    if len(rhs) != n or len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError('Некорректные размеры трёхдиагональной системы')
    
    c = upper[:]
    d = rhs[:]
    b = diag[:]
    
    for i in range(1, n):
        if abs(b[i - 1]) < 1e-15:
            raise ValueError('Нулевой диагональный элемент в методе прогонки')
        factor = lower[i - 1] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
    
    if abs(b[-1]) < 1e-15:
        raise ValueError('Нулевой диагональный элемент в методе прогонки')
    
    x = [0.0] * n
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        if abs(b[i]) < 1e-15:
            raise ValueError('Нулевой диагональный элемент в методе прогонки')
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    
    return x


def solve_boundary_value_problem(
    n: int = 100, 
    eps: float = 1e-8, 
    max_iter: int = 100, 
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Решение краевой задачи: y'' - y³ = x², y(0)=1, y(1)=3."""
    if n < 2:
        raise ValueError('Число отрезков должно быть не меньше 2')
    
    h = 1.0 / n
    x = [i * h for i in range(n + 1)]
    y = [1.0 + 2.0 * xi for xi in x]
    
    history = []
    interior_size = n - 1
    
    for iteration in range(1, max_iter + 1):
        lower = [1.0 / h**2] * (interior_size - 1)
        diag = []
        upper = [1.0 / h**2] * (interior_size - 1)
        rhs = []
        
        for i in range(1, n):
            residual = (y[i - 1] - 2.0 * y[i] + y[i + 1]) / h**2 - y[i] ** 3 - x[i] ** 2
            diag.append(-2.0 / h**2 - 3.0 * y[i] ** 2)
            rhs.append(-residual)
        
        delta = solve_tridiagonal(lower, diag, upper, rhs)
        
        max_delta = 0.0
        for i in range(1, n):
            y[i] += delta[i - 1]
            max_delta = max(max_delta, abs(delta[i - 1]))
        
        y[0] = 1.0
        y[n] = 3.0
        
        history.append({
            'iteration': iteration, 
            'max_delta': max_delta,
            'y': y.copy()
        })
        
        if max_delta < eps:
            break
    else:
        raise RuntimeError('Превышено максимальное число итераций в задаче 3')
    
    plot_path = None
    if output_dir is not None:
        from plotting import plot_task3_results_styled
        plot_path = plot_task3_results_styled(x, y, history, output_dir)
    
    return {
        'x': np.array(x),
        'y': np.array(y),
        'iterations': history[-1]['iteration'],
        'history': history,
        'plot_path': str(plot_path) if plot_path is not None else None,
    }
