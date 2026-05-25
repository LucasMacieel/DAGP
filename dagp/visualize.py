"""LON network plots and violin plots."""

from __future__ import annotations


import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import numpy as np

from dagp.lon import LONResult
from dagp.metrics import LONMetrics

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 10


def plot_lon(
    lon_result: LONResult,
    title: str = "",
    color: str = "steelblue",
    ax: plt.Axes = None,
    save_path: str = None,
) -> None:
    G = lon_result.graph
    if G.number_of_nodes() == 0:
        return

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()

    # Layout
    if G.number_of_nodes() <= 1:
        pos = {n: (0, 0) for n in G.nodes()}
    else:
        pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(G.number_of_nodes()))

    # Node sizes: proportional to number of searches that ended here
    basin_sizes = {}
    for result in lon_result.search_results:
        h = result.final_tree.tree_hash()
        basin_sizes[h] = basin_sizes.get(h, 0) + 1

    node_sizes = []
    node_colors = []
    for node in G.nodes():
        size = basin_sizes.get(node, 1) * 200 + 100
        node_sizes.append(min(size, 3000))  # cap size
        if node == lon_result.global_optimum_hash:
            node_colors.append("gold")
        else:
            node_colors.append(color)

    # Draw
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        alpha=0.4,
        edge_color="gray",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_axis_off()

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_lon_comparison(
    lon_no_scaling: LONResult,
    lon_linear_scaling: LONResult,
    equation_id: str,
    save_path: str = None,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_lon(
        lon_no_scaling,
        title=f"{equation_id} (no-scaling)",
        color="steelblue",
        ax=ax1,
    )
    plot_lon(
        lon_linear_scaling,
        title=f"{equation_id} (linear-scaling)",
        color="indianred",
        ax=ax2,
    )

    fig.suptitle(
        f"Local Optima Network — {equation_id}", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_violin_metrics(
    all_metrics: dict[str, tuple[LONMetrics, LONMetrics]],
    save_path: str = None,
) -> None:
    metric_names = ["nv", "ne", "C", "Cr", "avg_path_len", "pi", "S", "nhits"]
    metric_labels = ["nv", "ne", "C", "Cr", "l", "π", "S", "nhits"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (mname, mlabel) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[i]

        ns_vals = [getattr(m[0], mname) for m in all_metrics.values()]
        ls_vals = [getattr(m[1], mname) for m in all_metrics.values()]

        data = [ns_vals, ls_vals]
        parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showextrema=True)

        # Color the violins
        for pc, color in zip(parts["bodies"], ["steelblue", "indianred"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["no-scaling", "linear"])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "LON Metrics — Violin Plots (all equations)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
