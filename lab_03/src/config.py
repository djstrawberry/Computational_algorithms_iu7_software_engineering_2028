EPS = 1e-10

RANDOM_SEED = 42

ONE_D_CONFIG = {
    "nodes_count": 14,
    "x_min": -2.0,
    "x_max": 4.0,
    "noise_sigma": 0.35,
}

TWO_D_CONFIG = {
    "nodes_count": 90,
    "x_min": -2.0,
    "x_max": 2.5,
    "y_min": -2.0,
    "y_max": 2.0,
    "noise_sigma": 0.22,
}

MODEL_SELECTION_DATA = {
    "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "y": [10.5, 1.6, 0.55, 0.26, 0.15, 0.08],
    "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

BVP_CONFIG = {
    "x_start": 0.0,
    "x_end": 1.0,
    "y_start": 1.0,
    "y_end": 0.0,
    "grid_points": 401,
}


def build_paths(base_dir: str) -> dict:
    return {
        "base_dir": base_dir,
        "data_dir": f"{base_dir}/data",
        "results_dir": f"{base_dir}/results",
    }
