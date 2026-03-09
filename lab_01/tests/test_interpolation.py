from src.interpolation import get_nearest_nodes, compute_divided_differences, \
calculate_Newton_polynomial, interpolate_by_1D_Newton, interpolate_by_2D_Newton

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

# ---------------------------------------------
# Тест для функции interpolate_by_1D_Newton
# ---------------------------------------------
def test_interpolate_by_1D_Newton():
    x_args = [1, 2, 3]
    y_args = [1, 4, 9]

    interpolated_value = interpolate_by_1D_Newton(2.5, x_args, y_args, 2)

    assert abs(interpolated_value - 6.25) < EPS

# ---------------------------------------------
# Тест для функции interpolate_by_2D_Newton
# ---------------------------------------------
def test_interpolate_by_2D_Newton():
    T_args = [1, 2, 3]
    p_args = [1, 2, 3]
    values = [[1, 4, 9], [4, 16, 36], [9, 36, 81]]

    interpolated_value = interpolate_by_2D_Newton(2.5, 2.5, T_args, p_args, values, 2)

    assert abs(interpolated_value - 39.0625) < EPS