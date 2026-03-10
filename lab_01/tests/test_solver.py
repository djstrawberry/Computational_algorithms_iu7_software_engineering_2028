from src.solver import build_time_grid

#---------------------------------
# Тест для функции build_time_grid
#---------------------------------

def test_build_time_grid_round():
    t0 = 0.0
    tk = 1.0
    tau = 0.25

    expected_time_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    time_grid = build_time_grid(t0, tk, tau)

    assert time_grid == expected_time_grid

def test_build_time_grid_non_round():
    t0 = 0.0
    tk = 1.0
    tau = 0.3

    expected_time_grid = [0.0, 0.3, 0.6, 0.9]

    time_grid = build_time_grid(t0, tk, tau)

    assert time_grid == expected_time_grid