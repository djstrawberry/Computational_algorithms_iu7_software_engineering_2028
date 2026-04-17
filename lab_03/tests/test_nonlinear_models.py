import numpy as np

from nonlinear_models import fit_all_models


def test_fit_all_models_returns_sorted_by_mse():
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    y = np.array([10.5, 1.6, 0.55, 0.26, 0.15, 0.08])
    w = np.ones_like(x)

    models = fit_all_models(x, y, w)

    assert len(models) == 4
    assert models[0]["mse"] <= models[1]["mse"] <= models[2]["mse"] <= models[3]["mse"]
