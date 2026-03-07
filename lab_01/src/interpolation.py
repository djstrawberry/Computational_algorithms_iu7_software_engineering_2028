from .config import INTERPOLATION_DEGREES
import bisect

def interpolate_by_1D_Newton(x: float, x_args: list, y_args: list, degree: int) -> float:

    nodes_quantity = degree + 1
    nodes = get_nearest_nodes(x, x_args, nodes_quantity)

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
                
    return x_args[left + 1 : right]

        
def find_x_position(x: float, x_args: list) -> int:
    return bisect.bisect_left(x_args, x)