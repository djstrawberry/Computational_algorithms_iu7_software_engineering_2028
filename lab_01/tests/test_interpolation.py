from src.interpolation import get_nearest_nodes

def test_get_nearest_nodes_middle():
    x_values = [0, 2, 4, 6, 8]
    nodes = get_nearest_nodes(3, x_values, 3)

    assert len(nodes) == 3
    assert nodes == [0, 2, 4] or nodes == [2, 4, 6]


def test_get_nearest_nodes_left_edge():
    x_values = [0, 2, 4, 6, 8]
    nodes = get_nearest_nodes(-1, x_values, 3)

    assert nodes == [0, 2, 4]


def test_get_nearest_nodes_right_edge():
    x_values = [0, 2, 4, 6, 8]
    nodes = get_nearest_nodes(10, x_values, 3)

    assert nodes == [4, 6, 8]