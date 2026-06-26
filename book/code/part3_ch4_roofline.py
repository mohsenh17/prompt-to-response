import numpy as np
import matplotlib.pyplot as plt

def plot_roofline(
    peak_flops=1000,   # TFLOPs
    bandwidth=3.3,     # TB/s (approx H100 scale)
    ai_min=1e-2,
    ai_max=1e3
):
    """
    Clean roofline model visualization (publication / slide quality).
    """

    # Arithmetic intensity range
    ai = np.logspace(np.log10(ai_min), np.log10(ai_max), 600)

    # Roofline components
    memory_line = bandwidth * ai
    compute_line = np.full_like(ai, peak_flops)

    # Actual roofline (min of the two)
    roofline = np.minimum(memory_line, compute_line)

    # Ridge point
    ridge_ai = peak_flops / bandwidth

    fig, ax = plt.subplots(figsize=(9, 6))

    # ---- Fill regions (key improvement) ----
    ax.fill_between(
        ai,
        memory_line,
        compute_line,
        where=compute_line > memory_line,
        color="orange",
        alpha=0.15,
        label="Memory-bound region"
    )

    ax.fill_between(
        ai,
        compute_line,
        memory_line,
        where=memory_line > compute_line,
        color="skyblue",
        alpha=0.15,
        label="Compute-bound region"
    )

    # ---- Main lines ----
    ax.loglog(ai, memory_line, linestyle="--", linewidth=2, color="darkorange")
    ax.loglog(ai, compute_line, linewidth=2.5, color="navy")
    ax.loglog(ai, roofline, linewidth=3.5, color="black", label="Roofline")

    # ---- Ridge point ----
    ax.scatter([ridge_ai], [peak_flops], color="red", s=60, zorder=5)
    ax.annotate(
        f"Ridge point\nAI ≈ {300}",
        xy=(ridge_ai, peak_flops),
        xytext=(ridge_ai * 1.5, peak_flops * 0.2),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red"
    )

    # ---- Labels ----
    ax.set_title("Roofline Model (GPU Performance Limit)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=12)
    ax.set_ylabel("Performance (TFLOPs)", fontsize=12)

    # ---- Grid & style ----
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)

    # Cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("roofline_model.svg", dpi=300)

if __name__ == "__main__":
    plot_roofline()