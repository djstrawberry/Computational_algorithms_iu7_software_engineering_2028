import numpy as np
from matplotlib import pyplot as plt

from one_dimensional import predict_1d_polynomial
from two_dimensional import predict_2d_polynomial


def plot_1d_fits(table: dict, fit_results: list[dict], output_path: str) -> None:
    x = table["x"]
    y = table["y"]

    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 400)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#F8FBFE")
    ax.set_facecolor("#FCFEFF")

    ax.scatter(x, y, color="#244B7A", s=45, label="Табличные точки")

    colors = ["#D1495B", "#2A9D8F", "#7F5539", "#5E60CE"]
    for fit_result, color in zip(fit_results, colors):
        y_dense = predict_1d_polynomial(x_dense, fit_result)
        ax.plot(x_dense, y_dense, color=color, linewidth=2.2, label=f"n={fit_result['degree']}")

    ax.set_title("Одномерная аппроксимация методом наименьших квадратов")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_weight_impact(table: dict, fit_uniform: dict, fit_custom: dict, output_path: str) -> None:
    x = table["x"]
    y = table["y"]
    weights = table.get("weights", np.ones_like(x))
    # Масштабируем веса для размера точек (s):
    # min weight -> min size, max weight -> max size
    min_size, max_size = 40, 220
    w_min, w_max = np.min(weights), np.max(weights)
    if w_max > w_min:
        sizes = min_size + (weights - w_min) / (w_max - w_min) * (max_size - min_size)
    else:
        sizes = np.full_like(weights, (min_size + max_size) / 2)

    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 400)
    y_uniform = predict_1d_polynomial(x_dense, fit_uniform)
    y_custom = predict_1d_polynomial(x_dense, fit_custom)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#FFF9F0")
    ax.set_facecolor("#FFFCF6")

    scatter = ax.scatter(x, y, color="#4A4A4A", s=sizes, label="Точки одной таблицы (размер = вес)")
    ax.plot(x_dense, y_uniform, color="#247BA0", linewidth=2.2, label="rho=1, n=1")
    ax.plot(x_dense, y_custom, color="#F25F5C", linewidth=2.2, label="назначенные веса, n=1")

    ax.set_title("Влияние весов на положение аппроксимирующей прямой")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Добавим colorbar для визуализации весов (по желанию)
    # cbar = plt.colorbar(scatter, ax=ax)
    # cbar.set_label('Веса (rho)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_2d_fits(
    table: dict,
    fit_deg1: dict,
    fit_deg2: dict,
    output_path: str,
    show_interactive: bool = False,
) -> None:
    x = table["x"]
    y = table["y"]
    z = table["z"]

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 35)
    y_grid = np.linspace(float(np.min(y)), float(np.max(y)), 35)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    z_mesh_deg1 = predict_2d_polynomial(x_mesh, y_mesh, fit_deg1)
    z_mesh_deg2 = predict_2d_polynomial(x_mesh, y_mesh, fit_deg2)

    fig = plt.figure(figsize=(15, 6), facecolor="#F6FBF9")

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(x, y, z, color="#2F4858", s=20, alpha=0.85)
    ax1.plot_surface(x_mesh, y_mesh, z_mesh_deg1, alpha=0.5, cmap="viridis")
    ax1.set_title("Полином первой степени")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(x, y, z, color="#2F4858", s=20, alpha=0.85)
    ax2.plot_surface(x_mesh, y_mesh, z_mesh_deg2, alpha=0.5, cmap="plasma")
    ax2.set_title("Полином второй степени")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)

    if show_interactive:
        plt.show()

    plt.close(fig)


def plot_model_selection(x: np.ndarray, y: np.ndarray, models: list[dict], output_path: str) -> None:
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 300)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#F7F6FE")
    ax.set_facecolor("#FCFCFF")

    ax.scatter(x, y, color="#1D3557", s=55, label="Табличные точки")

    for model in models[:4]:
        if model["name"] == "a*x^b":
            y_dense = model["params"]["a"] * (x_dense ** model["params"]["b"])
        elif model["name"] == "a*exp(b*x)":
            y_dense = model["params"]["a"] * np.exp(model["params"]["b"] * x_dense)
        elif model["name"] == "a + b/x":
            y_dense = model["params"]["a"] + model["params"]["b"] / x_dense
        elif model["name"] == "x/(p + q*x)":
            y_dense = x_dense / (model["params"]["p"] + model["params"]["q"] * x_dense)
        else:
            continue

        ax.plot(x_dense, y_dense, linewidth=2.0, label=f"{model['name']}, mse={model['mse']:.4g}")

    ax.set_title("Сравнение нелинейных двухпараметрических моделей")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bvp_comparison(solution_m2: dict, solution_m3: dict, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#F7FBF7")
    ax.set_facecolor("#FCFFFC")

    ax.plot(solution_m2["x"], solution_m2["y"], color="#007F5F", linewidth=2.2, label="m=2")
    ax.plot(solution_m3["x"], solution_m3["y"], color="#BC4749", linewidth=2.2, label="m=3")

    ax.set_title("Сравнение приближенных решений краевой задачи")
    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
