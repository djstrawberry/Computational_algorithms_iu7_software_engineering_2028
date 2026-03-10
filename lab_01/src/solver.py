import math

from .model import get_model_params
from .interpolation import interpolate_by_1D_Newton
from .config import EPS

def solve_equation(params: dict, tables: dict) -> dict:
    
    time_grid = build_time_grid(params["t0"], params["tk"], params["tau"])
    results = init_results_dict()

    current_T = params["T0"]

    for current_t in time_grid:

        step_state = compute_step_state(current_t, current_T, params, tables)
        save_step_state(results, current_t, current_T, step_state)

        if not is_last_step(current_t, time_grid):
            
            current_T = make_Runge_Kutta_step(current_t, current_T, step_state, params, tables)

        compute_results(results, step_state, params)

    return results

def build_time_grid(t0: float, tk: float, tau: float) -> list[float]:

    time_grid = []
    n = 0

    while (t0 + n * tau <= tk + EPS):
        current_t = round(t0 + n * tau, 12)
        time_grid.append(current_t)
        n += 1

    return time_grid

def init_results_dict() -> dict:

    return {
        "t": [],
        "T": [],
        "p": [],
        "sigma": [],
        "q": [],
        "Rd": [],
        "Fr": []
    }

def compute_step_state(t: float, T: float, params: dict, tables: dict) -> dict:
    
    current_I = interpolate_by_1D_Newton(t, tables["I_t"]["x"], tables["I_t"]["y"], params["degree"])

    p, sigma, c, q = get_model_params(T, params["N_known"], tables, params["degree"], params["p_min"], params["p_max"])

    current_j = current_I / (math.pi * params["R"] ** 2)

    return {
        "I": current_I,
        "p": p,
        "sigma": sigma,
        "c": c,
        "q": q,
        "j": current_j
    }

def save_step_state(results: dict, t: float, T: float, step_state: dict) -> None:

    results["t"].append(t)
    results["T"].append(T)
    results["p"].append(step_state["p"])
    results["sigma"].append(step_state["sigma"])
    results["q"].append(step_state["q"])

def is_last_step(t: float, time_grid: list) -> bool:
    
    return abs(t - time_grid[-1]) < EPS

def make_Runge_Kutta_step(t: float, T: float, step_state: dict, params: dict, tables: dict) -> float:
    
    tau = params["tau"]

    k1 = compute_dT_dt(step_state)

    half_step_T = T + tau * k1 / 2
    half_step_t = t + tau / 2

    half_step_state = compute_step_state(half_step_t, half_step_T, params, tables)
    k2 = compute_dT_dt(half_step_state)

    next_T = T + tau * k2

    return next_T

def compute_dT_dt(step_state: dict) -> float:

    return (step_state["sigma"] * step_state["j"] ** 2- step_state["q"]) / step_state["c"]

def compute_results(results: dict, step_state: dict, params: dict) -> None:
    
    Rd = params["l"] / (math.pi * params["R"] ** 2 * step_state["sigma"])
    Fr = step_state["q"] * params["R"] ** 2

    results["Rd"].append(Rd)
    results["Fr"].append(Fr)