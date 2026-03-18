# utils.py
import numpy as np

def find_nearest_indices(arr, value):

    idx = np.searchsorted(arr, value)
    if idx == 0:
        return [0, 1]
    elif idx == len(arr):
        return [len(arr)-2, len(arr)-1]
    else:
        return [idx-1, idx]

def check_point_in_range(x, y, z, x0, y0, z0):

    if x0 < x[0] or x0 > x[-1]:
        print(f"Предупреждение: x={x0} вне диапазона [{x[0]}, {x[-1]}]")
        return False
    if y0 < y[0] or y0 > y[-1]:
        print(f"Предупреждение: y={y0} вне диапазона [{y[0]}, {y[-1]}]")
        return False
    if z0 < z[0] or z0 > z[-1]:
        print(f"Предупреждение: z={z0} вне диапазона [{z[0]}, {z[-1]}]")
        return False
    return True

def save_results(filename, results):
    """
    Сохраняет результаты в файл
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("РЕЗУЛЬТАТЫ ИНТЕРПОЛЯЦИИ\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Точка интерполяции: ({results['x0']}, {results['y0']}, {results['z0']})\n\n")
        
        f.write("Метод 1: Полиномы Ньютона\n")
        f.write(f"Степени: nx={results['nx']}, ny={results['ny']}, nz={results['nz']}\n")
        f.write(f"Результат: u = {results['newton_result']:.6f}\n\n")
        
        f.write("Метод 2: Сплайны\n")
        f.write(f"Результат: u = {results['spline_result']:.6f}\n\n")
        
        f.write("Метод 3: Смешанная интерполяция\n")
        f.write("Вариант А (сплайн по y):\n")
        f.write(f"Результат: u = {results['mixed_result_a']:.6f}\n")
        f.write("Вариант Б (сплайн по x):\n")
        f.write(f"Результат: u = {results['mixed_result_b']:.6f}\n\n")
        
        f.write("Сравнение с точным значением (аналитическая функция):\n")
        f.write(f"Точное значение: {results['exact_value']:.6f}\n")
        f.write(f"Погрешность (Ньютон): {abs(results['exact_value'] - results['newton_result']):.6f}\n")
        f.write(f"Погрешность (Сплайн): {abs(results['exact_value'] - results['spline_result']):.6f}\n")    