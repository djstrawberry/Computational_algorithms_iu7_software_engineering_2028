import numpy as np
from src.data import load_data
from src.newton_interpolation import newton_interpolate_1d, divided_differences
from src.spline_interpolation import spline_interpolate_1d

def test_1d_interpolation():
 
    print("\n" + "=" * 40)
    print("ТЕСТ ОДНОМЕРНОЙ ИНТЕРПОЛЯЦИИ")
    print("=" * 40)
    
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])  # x^2
    
    x_test = 2.5
    exact = x_test**2
    
    newton_val = newton_interpolate_1d(x, y, x_test)
    
    spline_val = spline_interpolate_1d(x, y, x_test)
    
    print(f"x = {x_test}, точное значение: {exact}")
    print(f"Полином Ньютона: {newton_val:.6f}")
    print(f"Сплайн: {spline_val:.6f}")

def test_divided_differences():

    print("\n" + "=" * 40)
    print("ТЕСТ РАЗДЕЛЕННЫХ РАЗНОСТЕЙ")
    print("=" * 40)
    
    x = np.array([0, 1, 2])
    y = np.array([1, 2, 3])
    
    diff = divided_differences(x, y)
    print(f"Разделенные разности для линейной функции: {diff}")
    print("Должны быть: [1, 1, 0]")

if __name__ == "__main__":
    test_1d_interpolation()
    test_divided_differences()