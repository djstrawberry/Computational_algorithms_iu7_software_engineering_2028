# main.py (обновленная точная функция)
import numpy as np
import os

from data import load_data, print_table
from newton_interpolation import newton_interpolate_3d
from spline_interpolation import spline_interpolate_3d
from mixed_interpolation import mixed_interpolate_3d
from utils import check_point_in_range, save_results

def exact_function(x, y, z):

    return x**2 + y**2 + z**2

def main():

    os.makedirs('results', exist_ok=True)
    
    x, y, z, u = load_data()
    
    print_table(u, x, y, z)
    
    x0 = 2.5  # Точка для интерполяции по x
    y0 = 2.3  # по y
    z0 = 1.7  # по z
    
    # Степени полиномов Ньютона (максимум 4, т.к. 5 узлов)
    nx = 3
    ny = 3
    nz = 2
    
    print("\n" + "=" * 60)
    print("ПАРАМЕТРЫ ИНТЕРПОЛЯЦИИ")
    print("=" * 60)
    print(f"Точка интерполяции: x = {x0}, y = {y0}, z = {z0}")
    print(f"Степени полиномов: nx = {nx}, ny = {ny}, nz = {nz}")
    
    if not check_point_in_range(x, y, z, x0, y0, z0):
        print("Точка вне диапазона! Результаты могут быть неточными.")
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ИНТЕРПОЛЯЦИИ")
    print("=" * 60)
    
    # Метод 1: Полиномы Ньютона
    newton_result = newton_interpolate_3d(x, y, z, u, x0, y0, z0, nx, ny, nz)
    print(f"\nМетод 1 (Полиномы Ньютона):")
    print(f"u({x0}, {y0}, {z0}) = {newton_result:.6f}")
    
    # Метод 2: Сплайны
    spline_result = spline_interpolate_3d(x, y, z, u, x0, y0, z0)
    print(f"\nМетод 2 (Сплайны):")
    print(f"u({x0}, {y0}, {z0}) = {spline_result:.6f}")
    
    # Метод 3: Смешанная интерполяция
    mixed_result_a = mixed_interpolate_3d(x, y, z, u, x0, y0, z0, ny_spline=True)
    mixed_result_b = mixed_interpolate_3d(x, y, z, u, x0, y0, z0, ny_spline=False)
    
    print(f"\nМетод 3 (Смешанная):")
    print(f"Вариант А (сплайн по y): u = {mixed_result_a:.6f}")
    print(f"Вариант Б (сплайн по x): u = {mixed_result_b:.6f}")
    
    # Точное значение (для проверки)
    exact = exact_function(x0, y0, z0)
    print(f"\nТочное значение (аналитическое): u = {exact:.6f}")
    print(f"(По таблице видно, что функция u = x² + y² + z²)")
    
    # Погрешности
    print(f"\nПогрешности:")
    print(f"Ньютон: {abs(exact - newton_result):.6f}")
    print(f"Сплайн: {abs(exact - spline_result):.6f}")
    print(f"Смешанная А: {abs(exact - mixed_result_a):.6f}")
    print(f"Смешанная Б: {abs(exact - mixed_result_b):.6f}")
    
    results = {
        'x0': x0, 'y0': y0, 'z0': z0,
        'nx': nx, 'ny': ny, 'nz': nz,
        'newton_result': newton_result,
        'spline_result': spline_result,
        'mixed_result_a': mixed_result_a,
        'mixed_result_b': mixed_result_b,
        'exact_value': exact
    }
    
    save_results('results/output.txt', results)
    print("\nРезультаты сохранены в файл 'results/output.txt'")

if __name__ == "__main__":
    main()