"""Главный модуль лабораторной работы."""

import numpy as np
from task1 import solve_task1
from task2 import solve_task2
from task3 import solve_boundary_value_problem
from plotting import ensure_results_dir


def main():
    """Точка входа."""
    results_dir = ensure_results_dir()

    # Задача 1
    results_1 = solve_task1()
    
    # Задача 2
    target_phi = 0.5
    x2, history_2, iters_2 = solve_task2(target_phi=target_phi)

    # Задача 3
    n = 100
    result_3 = solve_boundary_value_problem(
        n=n, 
        eps=1e-8, 
        max_iter=100, 
        output_dir=results_dir
    )
    
    x = result_3['x']
    y = result_3['y']
    iterations = result_3['iterations']
    
    # Сохранение численных результатов
    save_results_to_file(results_1, x2, target_phi, x, y, n, iterations, results_dir)


def save_results_to_file(results_1, x2, target_phi, x, y, n, iterations, results_dir):
    """Сохранение численных результатов в файл."""
    import os
    
    filepath = os.path.join(results_dir, "numerical_results.txt")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ЛАБОРАТОРНОЙ РАБОТЫ\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Задача 1. Решение системы нелинейных уравнений\n")
        f.write("-" * 40 + "\n")
        for r in results_1:
            f.write(f"Допуск: {r['tol']:.0e}\n")
            f.write(f"x = {r['x']:.10f}, y = {r['y']:.10f}\n")
            f.write(f"Итераций: {r['iterations']}\n\n")
        
        f.write("Задача 2. Аргумент функции Лапласа\n")
        f.write("-" * 40 + "\n")
        f.write(f"Φ(x) = {target_phi}\n")
        f.write(f"x = {x2:.10f}\n\n")
        
        f.write("Задача 3. Краевая задача\n")
        f.write("-" * 40 + "\n")
        f.write(f"n = {n}, итераций: {iterations}\n")
        f.write("x      y(x)\n")
        for i in range(0, n + 1, n // 10):
            f.write(f"{x[i]:.3f}  {y[i]:.8f}\n")


if __name__ == "__main__":
    main()
