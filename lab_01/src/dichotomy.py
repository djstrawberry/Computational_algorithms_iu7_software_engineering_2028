from .interpolation import interpolate_by_2D_Newton

EPS = 1e-6

def find_pressure_by_dichotomy(T: float, N_known: float, T_args: list, p_args: list, N_values: list, degree: int, p_min: float, p_max: float) -> float:

    while abs(p_max - p_min) > EPS:
        p_median = (p_max + p_min) / 2
        f_median = N_difference(p_median, T, T_args, p_args, N_values, degree, N_known)
        f_min = N_difference(p_min, T, T_args, p_args, N_values, degree, N_known)

        if abs(f_median) < EPS:
            return p_median

        if f_median * f_min < 0:
            p_max = p_median
        else:
            p_min = p_median

    return (p_max + p_min) / 2

def N_difference(p: float, T: float, T_args: list, p_args: list, N_values: list, degree: int,N_known: float) -> float:
   
    N_interpolated = interpolate_by_2D_Newton(T, p, T_args, p_args, N_values, degree)

    return N_interpolated - N_known 