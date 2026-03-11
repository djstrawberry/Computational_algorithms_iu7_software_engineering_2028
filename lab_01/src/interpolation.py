import bisect

def interpolate_by_1D_Newton(x: float, x_args: list, y_args: list, degree: int) -> float:

    x_clamped = max(x_args[0], min(x, x_args[-1]))
    
    nodes_quantity = degree + 1
    nodes = get_nearest_nodes(x_clamped, x_args, nodes_quantity)
    x_local = [x_args[i] for i in nodes]
    y_local = [y_args[i] for i in nodes]
    divided_differences = compute_divided_differences(x_local, y_local)

    return calculate_Newton_polynomial(x_clamped, x_local, divided_differences)

def interpolate_by_2D_Newton(T: float, p: float, T_args: list, p_args: list, values: list,
                             T_degree: int, p_degree: int) -> float:

    T_clamped = max(T_args[0], min(T, T_args[-1]))
    p_clamped = max(p_args[0], min(p, p_args[-1]))

    T_nodes = get_nearest_nodes(T_clamped, T_args, T_degree + 1)
    values_at_p = []
    T_local = []
    for i in T_nodes:
        T_local.append(T_args[i])
        row_values = values[i]

        values_at_p.append(interpolate_by_1D_Newton(p_clamped, p_args, row_values, p_degree))

    return interpolate_by_1D_Newton(T_clamped, T_local, values_at_p, T_degree)

def calculate_Newton_polynomial(x: float, x_args: list, coefficients: list) -> float:

    polynomial_value = 0
    product = 1

    for i in range(len(coefficients)):
        polynomial_value += coefficients[i] * product
        product *= x - x_args[i]

    return polynomial_value

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
    
    if nodes_quantity == len(x_args):
        return list(range(len(x_args)))
    
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