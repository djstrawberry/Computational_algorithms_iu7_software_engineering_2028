from src.interpolation import get_nearest_nodes, compute_divided_differences, calculate_Newton_polynomial
EPS = 1e-10

# -----------------------------------
# Тесты для функции get_nearest_nodes
# -----------------------------------
def test_get_nearest_nodes_middle():
    x_values = [0, 2, 4, 6, 8]
    nodes = get_nearest_nodes(3, x_values, 3)

    assert len(nodes) == 3
    assert nodes == [0, 1, 2] or nodes == [1, 2, 3]


def test_get_nearest_nodes_left_edge():
    x_values = [0, 2, 4, 6, 8]
    nodes = get_nearest_nodes(-1, x_values, 3)

    assert nodes == [0, 1, 2]


def test_get_nearest_nodes_right_edge():
    x_values = [0, 2, 4, 6, 8]
    nodes = get_nearest_nodes(10, x_values, 3)

    assert nodes == [2, 3, 4]

# ---------------------------------------------
# Тесты для функции compute_divided_differences
# ---------------------------------------------
def test_compute_divided_differences():
    x_args = [1, 2, 3]
    y_args = [1, 4, 9]

    divided_differences = compute_divided_differences(x_args, y_args)

    assert len(divided_differences) == len(x_args)
    assert divided_differences[0] == 1
    assert divided_differences[1] == 3
    assert divided_differences[2] == 1

# ---------------------------------------------
# Тесты для функции calculate_Newton_polynomial
# ---------------------------------------------
def test_calculate_Newton_polynomial():
    x_args = [1, 2, 3]
    coefficients = [1, 3, 1]

    polynomial_value = calculate_Newton_polynomial(2.5, x_args, coefficients)

    assert abs(polynomial_value - 6.25) < EPS