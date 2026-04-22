"""Оформление графиков в заданном стиле."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os
from pathlib import Path


def ensure_results_dir() -> str:
    """Создаёт папку results, если её нет, и возвращает путь к ней."""
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def setup_plot_style():
    """Настройка стиля графиков."""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def plot_task3_results_styled(
    x: List[float],
    y: List[float],
    history: List[Dict],
    output_dir: str
) -> Path:
    """Построение графика y(x) для задачи 3 в заданном стиле."""
    setup_plot_style()

    colors = {
        "T": "#C2185B",
        "p": "#2E7D32",
        "sigma": "#E91E63",
    }

    figure_bg = "#FFF8FB"
    axes_bg = "#FFFDFE"
    grid_color = "#EADFE6"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=figure_bg)
    ax.set_facecolor(axes_bg)

    ax.plot(x, y, color=colors["T"], linewidth=2.2, label='Численное решение y(x)')

    x_arr = np.array(x)
    y_initial = 1.0 + 2.0 * x_arr
    ax.plot(x, y_initial, color=colors["p"], linewidth=1.5,
            linestyle='--', alpha=0.7, label='Начальное приближение')

    ax.scatter([0, 1], [1, 3], color=colors["sigma"], s=80, zorder=5, 
               label='Граничные условия')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Решение краевой задачи $y\'\' - y^3 = x^2$', 
                 fontsize=14, color="#3E2D36", fontweight="semibold")

    ax.grid(True, color=grid_color, alpha=0.8, linewidth=0.8)
    ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='#BFAEB8')

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color("#BFAEB8")

    ax.tick_params(colors="#5A4B53")
    ax.title.set_color("#4C3A43")

    plt.tight_layout()

    results_dir = Path(output_dir) if output_dir else Path(ensure_results_dir())
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "task3_solution.png"
    
    plt.savefig(save_path, dpi=150, facecolor=figure_bg, bbox_inches='tight')
    plt.close()
    
    return save_path
