from matplotlib import pyplot as plt

def plot_results(results):
    t = results["t"]

    colors = {
        "T": "#C2185B",      # raspberry pink
        "p": "#2E7D32",      # deep green
        "sigma": "#E91E63",  # bright pink
        "q": "#66BB6A",      # soft green
        "Rd": "#F48FB1",     # pastel pink
        "Fr": "#388E3C",     # balanced green
    }

    figure_bg = "#FFF8FB"
    axes_bg = "#FFFDFE"
    grid_color = "#EADFE6"
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor=figure_bg)
    fig.suptitle('Зависимости параметров от времени', fontsize=16, color="#3E2D36", fontweight="semibold")
    
    axes[0, 0].set_facecolor(axes_bg)
    axes[0, 0].plot(t, results["T"], color=colors["T"], linewidth=2.2)
    axes[0, 0].set_title("T(t)")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("T")
    axes[0, 0].grid(True, color=grid_color, alpha=0.8, linewidth=0.8)
    
    axes[0, 1].set_facecolor(axes_bg)
    axes[0, 1].plot(t, results["p"], color=colors["p"], linewidth=2.2)
    axes[0, 1].set_title("p(t)")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("p")
    axes[0, 1].grid(True, color=grid_color, alpha=0.8, linewidth=0.8)
    
    axes[0, 2].set_facecolor(axes_bg)
    axes[0, 2].plot(t, results["sigma"], color=colors["sigma"], linewidth=2.2)
    axes[0, 2].set_title("σ(t)")
    axes[0, 2].set_xlabel("t")
    axes[0, 2].set_ylabel("σ")
    axes[0, 2].grid(True, color=grid_color, alpha=0.8, linewidth=0.8)
    
    axes[1, 0].set_facecolor(axes_bg)
    axes[1, 0].plot(t, results["q"], color=colors["q"], linewidth=2.2)
    axes[1, 0].set_title("q(t)")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("q")
    axes[1, 0].grid(True, color=grid_color, alpha=0.8, linewidth=0.8)
    
    axes[1, 1].set_facecolor(axes_bg)
    axes[1, 1].plot(t, results["Rd"], color=colors["Rd"], linewidth=2.2)
    axes[1, 1].set_title("Rd(t)")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("Rd")
    axes[1, 1].grid(True, color=grid_color, alpha=0.8, linewidth=0.8)
    
    axes[1, 2].set_facecolor(axes_bg)
    axes[1, 2].plot(t, results["Fr"], color=colors["Fr"], linewidth=2.2)
    axes[1, 2].set_title("Fr(t)")
    axes[1, 2].set_xlabel("t")
    axes[1, 2].set_ylabel("Fr")
    axes[1, 2].grid(True, color=grid_color, alpha=0.8, linewidth=0.8)

    for row in axes:
        for ax in row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#BFAEB8")
            ax.spines["bottom"].set_color("#BFAEB8")
            ax.tick_params(colors="#5A4B53")
            ax.title.set_color("#4C3A43")
    
    plt.tight_layout()
    plt.show()
