"""
Generate publication-quality figures from experiment results.

Usage:
    python plot_figures.py

Requires: matplotlib, numpy
Input:  figures/experiment1.json, figures/experiment2.json
Output: figures/rate_comparison.pdf, figures/bias_variance_decomp.pdf
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "text.usetex": True,
    "figure.figsize": (5.5, 4),
    "figure.dpi": 300,
})


def plot_rate_comparison(fig_dir):
    """Figure 1: Log-log plot of operator norm error vs n (4 estimators)."""
    with open(os.path.join(fig_dir, "experiment1.json")) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for idx, alpha in enumerate([0.55, 0.6, 0.7]):
        ax = axes[idx]
        ns, orig, impr, bi, reg = [], [], [], [], []
        for key, val in data.items():
            if val["alpha"] == alpha:
                ns.append(val["n"])
                orig.append(val["original_mean"])
                impr.append(val["improved_mean"])
                bi.append(val["burnin_mean"])
                reg.append(val["regression_mean"])

        order = np.argsort(ns)
        ns_arr = np.array(ns, dtype=float)[order]
        orig_arr = np.array(orig)[order]
        impr_arr = np.array(impr)[order]
        bi_arr = np.array(bi)[order]
        reg_arr = np.array(reg)[order]

        ax.loglog(ns_arr, orig_arr, "o-",
                  label=r"BM orig ($\beta^*$, $\rho\!=\!1$)",
                  color="C0", markersize=4, linewidth=1.3)
        ax.loglog(ns_arr, impr_arr, "^-",
                  label=r"BM opt ($\beta^\dagger$, $\rho\!=\!1$)",
                  color="C2", markersize=4, linewidth=1.3)
        ax.loglog(ns_arr, bi_arr, "s-",
                  label=r"BM opt ($\beta^\dagger$, $\rho\!=\!0.5$)",
                  color="C1", markersize=4, linewidth=1.3)
        ax.loglog(ns_arr, reg_arr, "D-",
                  label=r"Traj.\ regression",
                  color="C3", markersize=4, linewidth=1.3)

        # Reference slopes
        rate1 = (1 - alpha) / 4
        rate2 = (1 - alpha) / 3
        rate3 = (1 - alpha) / 2
        mid = len(ns_arr) // 2

        c1 = orig_arr[mid] / ns_arr[mid] ** (-rate1)
        ax.loglog(ns_arr, c1 * ns_arr ** (-rate1), ":", color="C0",
                  alpha=0.5, linewidth=1,
                  label=f"$n^{{-{rate1:.3f}}}$")

        c2 = impr_arr[mid] / ns_arr[mid] ** (-rate2)
        ax.loglog(ns_arr, c2 * ns_arr ** (-rate2), ":", color="C2",
                  alpha=0.5, linewidth=1,
                  label=f"$n^{{-{rate2:.3f}}}$")

        c3 = reg_arr[mid] / ns_arr[mid] ** (-rate3)
        ax.loglog(ns_arr, c3 * ns_arr ** (-rate3), ":", color="C3",
                  alpha=0.5, linewidth=1,
                  label=f"$n^{{-{rate3:.3f}}}$")

        # Empirical slopes (last 3 points)
        if len(ns_arr) >= 3:
            sl_orig = np.polyfit(np.log10(ns_arr[-3:]),
                                 np.log10(orig_arr[-3:]), 1)[0]
            sl_impr = np.polyfit(np.log10(ns_arr[-3:]),
                                 np.log10(impr_arr[-3:]), 1)[0]
            sl_bi = np.polyfit(np.log10(ns_arr[-3:]),
                               np.log10(bi_arr[-3:]), 1)[0]
            sl_reg = np.polyfit(np.log10(ns_arr[-3:]),
                                np.log10(reg_arr[-3:]), 1)[0]
            ax.text(0.03, 0.03,
                    f"slopes (tail 3):\n"
                    f"orig: {sl_orig:.3f}\n"
                    f"opt: {sl_impr:.3f}\n"
                    f"bi: {sl_bi:.3f}\n"
                    f"reg: {sl_reg:.3f}",
                    transform=ax.transAxes, fontsize=6.5,
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="wheat", alpha=0.5))

        ax.set_xlabel("$n$")
        ax.set_title(f"$\\alpha = {alpha}$")
        ax.legend(fontsize=6, loc="upper right", ncol=1)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(
        r"$\|\widehat{V}_n - V\|_{\mathrm{op}}$"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rate_comparison.pdf"),
                bbox_inches="tight")
    plt.close()
    print("  Saved rate_comparison.pdf")


def plot_bias_variance(fig_dir):
    """Figure 2: Per-block bias and variance."""
    with open(os.path.join(fig_dir, "experiment2.json")) as f:
        data = json.load(f)

    blocks = [d["block"] + 1 for d in data]  # 1-indexed
    biases = [d["bias"] for d in data]
    variances = [d["variance"] for d in data]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(blocks, biases, "o-", label="Bias", markersize=4,
                linewidth=1.2, color="C0")
    ax.semilogy(blocks, variances, "s-", label="Variance", markersize=4,
                linewidth=1.2, color="C1")
    cutoff = len(blocks) // 2
    ax.axvline(x=cutoff, color="red", linestyle=":", alpha=0.7,
               linewidth=1.5, label=f"Burn-in cutoff ($\\rho=0.5$)")
    ax.set_xlabel("Block index $m$")
    ax.set_ylabel("Operator norm")
    ax.set_title(
        r"Per-block bias and variance ($\alpha=0.55$, $n=10^5$, $\beta=\beta^\dagger$)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "bias_variance_decomp.pdf"),
                bbox_inches="tight")
    plt.close()
    print("  Saved bias_variance_decomp.pdf")


if __name__ == "__main__":
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")

    print("Plotting rate comparison...")
    plot_rate_comparison(fig_dir)

    print("Plotting bias-variance decomposition...")
    plot_bias_variance(fig_dir)

    print("Done. Figures saved to figures/")
