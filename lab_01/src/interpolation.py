from .config import INTERPOLATION_DEGREES
import bisect

def calculate_Newton_polynomial(x: float, x_args: list, coefficients: list) -> float:

    polynomial_value = 0
    product = 1

    for i in range(len(coefficients)):
        polynomial_value += coefficients[i] * product
        product *= x - x_args[i]

    return polynomial_value

def interpolate_by_1D_Newton(x: float, x_args: list, y_args: list, degree: int) -> float:

    nodes_quantity = degree + 1
    nodes = get_nearest_nodes(x, x_args, nodes_quantity)
    x_local = [x_args[i] for i in nodes]
    y_local = [y_args[i] for i in nodes]
    divided_differences = compute_divided_differences(x_local, y_local)
    return calculate_Newton_polynomial(x, x_local, divided_differences)


def compute_divided_differences(x_args: list, y_args: list) -> list:

    if (len(x_args) != len(y_args)):
        raise ValueError("Количество аргументов x и y должно быть одинаковым.")
    
    divided_differences = y_args.copy()
    args_length = len(x_args)

    for i in range(args_length):
        for j in range(args_length - 1, i, -1):
            divided_differences[j] = (divided_differences[j] - divided_differences[j - 1]) / (x_args[j] - x_args[j - i - 1])
    
    return divided_differences


def get_nearest_nodes(x: float, x_args: list, nodes_quantity: int) -> list:

    if (nodes_quantity > len(x_args)):
        raise ValueError("Количество узлов должно быть меньше количества точек в таблице.")
    
    right = find_x_position(x, x_args)
    left = right - 1

    while (right - left - 1 < nodes_quantity):
        if left < 0:
            right += 1
        elif right >= len(x_args):
            left -= 1
        else:
            if (abs(x - x_args[left]) <= abs(x - x_args[right])):
                left -= 1
            else:
                right += 1
                
    return list(range(left + 1, right))

        
def find_x_position(x: float, x_args: list) -> int:
    return bisect.bisect_left(x_args, x)