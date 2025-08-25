import os
import numpy as np
from rules.rules import FExp
import matplotlib.pyplot as plt


def alpha_levels_gradient(alphas, out_path="results/figures/paper",
                          t_step=5, x_step=0.02, levels=20, cmap="Blues_r"):
    """
    Saves one grayscale heatmap per alpha with paper-friendly styling.
    """
    os.makedirs(out_path, exist_ok=True)

    # Mesh
    t_values = np.arange(0, 100 + t_step, t_step)
    x_values = np.arange(0, 1 + x_step, x_step)
    T, Cs = np.meshgrid(t_values, x_values)

    # Global style (serif + slightly larger)
    plt.rcParams.update({
        "font.size": 14,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral"
    })

    for alpha in alphas:
        # Evaluate F_exp on grid
        Z = np.empty_like(T, dtype=float)
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                Z[i, j] = FExp(alpha=alpha, x=Cs[i, j], t=T[i, j])

        # Figure
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        # Soft, formal shading
        h = ax.contourf(T, Cs, Z, levels=levels, cmap=cmap)
        # Very light isolines for structure
        ax.contour(T, Cs, Z, levels=10, colors="k", linewidths=0.35, alpha=0.2)

        # Axes + labels (math-style)
        ax.set_xlabel(r"t")
        ax.set_ylabel(r"x")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)

        # Clean frame
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(True, which="both", alpha=0.15, linewidth=0.3)

        # Colorbar on the right
        cbar = fig.colorbar(h, ax=ax, fraction=0.048, pad=0.04, ticks=np.arange(0, 1.01, 0.1))
        cbar.set_label(rf"$F^{{exp}}_{{\alpha={alpha}}}$")
        cbar.outline.set_linewidth(0.4)


        # Save
        fname = f"Figure_{alpha}.png"
        fig.savefig(os.path.join(out_path, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {os.path.join(out_path, fname)}")



if __name__ == '__main__':
    # dynamic Fexp score as a function of alpha value
    alpha_levels_gradient(alphas=[0.01, 0.05, 0.1, 0.15])
