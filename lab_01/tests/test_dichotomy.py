from src.dichotomy import find_pressure_by_dichotomy

EPS = 1e-10

# ---------------------------------------------
# Тест для функции find_pressure_by_dichotomy
# ---------------------------------------------
def test_find_pressure_by_dichotomy():
    T = 2.0
    expected_p = 2.5
    N_known = 25.0

    T_args = [1.0, 2.0, 3.0]
    p_args = [1.0, 2.0, 3.0]
    N_values = [
        [1.0, 4.0, 9.0],
        [4.0, 16.0, 36.0],
        [9.0, 36.0, 81.0]
    ]

    p_min = 1.0
    p_max = 3.0

    found_pressure = find_pressure_by_dichotomy(
        T=T,
        N_known=N_known,
        T_args=T_args,
        p_args=p_args,
        N_values=N_values,
        T_degree=2,
        p_degree=2,
        p_min=p_min,
        p_max=p_max
    )

    assert abs(found_pressure - expected_p) < EPS