from .interpolation import interpolate_by_2D_Newton
from .dichotomy import find_pressure_by_dichotomy

def get_model_params(T: float, N_known: float, tables: dict, degree: int, p_min: float, p_max: float) -> tuple[float, float, float, float]:

    p = find_pressure_by_dichotomy(T, N_known, tables["Nh"]["T"], tables["Nh"]["p"], tables["Nh"]["values"], degree, p_min, p_max)

    sigma_value = interpolate_by_2D_Newton(T, p, tables["sigma"]["T"], tables["sigma"]["p"], tables["sigma"]["values"], degree)
    c_value = interpolate_by_2D_Newton(T, p, tables["c"]["T"], tables["c"]["p"], tables["c"]["values"], degree)
    q_value = interpolate_by_2D_Newton(T, p, tables["q"]["T"], tables["q"]["p"], tables["q"]["values"], degree)

    return p, sigma_value, c_value, q_value