"""
visualization/visualization.py
Author: Param Kapur
Date: 5/10/25

Visualization tools for fair hierarchical clustering.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any

from models.hierarchy import Hierarchy
from models.node import Node


def plot_colored_tree(
    hierarchy: Hierarchy,
    color_ids: List[List[int]],
    title: str = "Hierarchical Clustering",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    leaf_fontsize: int = 8,
    linewidth: float = 1.4,
) -> None:
    """
    Visualise a hierarchical clustering as a dendrogram with coloured leaves.

    Args
    ----
    hierarchy   : Hierarchy – tree to plot (must have root)
    color_ids   : list[list[int]] – each inner list holds point‑ids that share a colour
    title       : str – figure title
    figsize     : tuple(int, int) – matplotlib figure size
    save_path   : str|None – if given, figure is written there and closed
    leaf_fontsize : int – fontsize for leaf labels
    linewidth     : float – thickness of branch lines
    """
    if hierarchy.is_empty():
        raise ValueError("Hierarchy is empty – nothing to plot")

    #########################################################
    # 1.  Pre‑compute geometry: x‑coordinate for every leaf #
    #########################################################
    leaves: List[Node] = hierarchy.get_leaves()
    n_leaves = len(leaves)
    if n_leaves == 0:
        raise ValueError("Hierarchy has no leaves")

    # map leaf‑node.id -> x‑position (evenly spaced integers)
    leaf_xpos = {leaf.id: idx for idx, leaf in enumerate(leaves)}

    # height of each node = depth from root (root at depth 0)
    def depth(node: Node) -> int:
        d = 0
        p = node.parent
        while p is not None:
            d += 1
            p = p.parent
        return d

    max_depth = max(depth(leaf) for leaf in leaves)

    #########################################################
    # 2.  Recursive drawing                                #
    #########################################################

    # palette (repeat if user gives > len(default))
    base_colors = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#9467bd",  # purple
        "#ff7f0e",  # orange
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]
    palette = (
        base_colors
        if len(color_ids) <= len(base_colors)
        else base_colors * ((len(color_ids) // len(base_colors)) + 1)
    )

    id_to_colour = {}
    for colour_idx, ids in enumerate(color_ids):
        for pt_id in ids:
            id_to_colour[pt_id] = palette[colour_idx]

    fig, ax = plt.subplots(figsize=figsize)

    def plot_subtree(node: Node) -> Tuple[float, float]:
        """
        Draw subtree rooted at *node*.
        Returns (x_root, y_root) location of this node.
        """
        if node.is_leaf():
            x = leaf_xpos[node.id]
            y = max_depth  # leaves at bottom
            colour = id_to_colour.get(node.id, "#7f7f7f")  # default grey
            ax.plot(x, y, "o", color=colour, markersize=6, zorder=3)
            ax.text(
                x,
                y + 0.1,
                str(node.id),
                ha="center",
                va="top",
                fontsize=leaf_fontsize,
                rotation=45,
            )
            return x, y

        # Plot children first
        child_coords = [plot_subtree(c) for c in node.children]
        xs, ys = zip(*child_coords)

        # Internal node’s y coordinate is one level above its deepest child
        y_node = min(ys) - 1
        x_node = (min(xs) + max(xs)) / 2

        # Horizontal line connecting children
        ax.plot([min(xs), max(xs)], [y_node, y_node], color="k", lw=linewidth)

        # Vertical line down to each child
        for cx, cy in child_coords:
            ax.plot([cx, cx], [cy, y_node], color="k", lw=linewidth)

        return x_node, y_node

    plot_subtree(hierarchy.root)

    #########################################################
    # 3.  Aesthetics                                        #
    #########################################################
    ax.set_title(title, pad=20)
    ax.set_xlabel("")  # leaves already labelled
    ax.set_ylabel("Tree Level (root = 0)")

    # Flip y‑axis so root at top
    ax.set_ylim(max_depth + 0.5, -1)
    ax.set_xlim(-1, n_leaves)

    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # build legend
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette[i],
            markersize=8,
            label=f"Colour {i}",
        )
        for i in range(len(color_ids))
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_balance_distribution(
    balances: List[float],
    data_balance: float,
    title: str = "Cluster Balance Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the distribution of color balances across clusters.

    Args:
        balances: List of balance values (proportion of blue points)
        data_balance: The global balance of the dataset
        title: Plot title
        figsize: Figure size
        save_path: If provided, save the figure to this path
    """
    plt.figure(figsize=figsize)

    # Plot histogram with KDE
    sns.histplot(balances, kde=True, bins=20, alpha=0.7)

    # Add vertical line for global balance
    plt.axvline(
        x=data_balance,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Global Balance: {data_balance:.2f}",
    )

    # Set labels and title
    plt.xlabel("Proportion of Blue Points")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()

    # Adjust layout and save/show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_comparison(
    vanilla_balances: List[float],
    fair_balances: List[float],
    data_balance: float,
    title: str = "Balance Comparison: Vanilla vs Fair",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot side-by-side comparison of vanilla and fair balance distributions.

    Args:
        vanilla_balances: Balance values for vanilla clustering
        fair_balances: Balance values for fair clustering
        data_balance: The global balance of the dataset
        title: Plot title
        figsize: Figure size
        save_path: If provided, save the figure to this path
    """
    plt.figure(figsize=figsize)

    # Create subplots
    plt.subplot(1, 2, 1)
    sns.histplot(vanilla_balances, kde=True, bins=15, alpha=0.7)
    plt.axvline(x=data_balance, color="r", linestyle="--", linewidth=2)
    plt.title("Vanilla Clustering")
    plt.xlabel("Proportion of Blue Points")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.histplot(fair_balances, kde=True, bins=15, alpha=0.7)
    plt.axvline(
        x=data_balance,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Global Balance: {data_balance:.2f}",
    )
    plt.title("Fair Clustering")
    plt.xlabel("Proportion of Blue Points")
    plt.ylabel("Frequency")
    plt.legend()

    # Add overall title
    plt.suptitle(title, fontsize=16)

    # Adjust layout and save/show
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_cost_vs_size(
    sizes: List[int],
    cost_ratios: List[float],
    title: str = "Cost Ratio vs Sample Size",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the cost ratio vs sample size.

    Args:
        sizes: List of sample sizes
        cost_ratios: List of cost ratios (fair cost / vanilla cost)
        title: Plot title
        figsize: Figure size
        save_path: If provided, save the figure to this path
    """
    plt.figure(figsize=figsize)

    # Plot line with markers
    plt.plot(sizes, cost_ratios, "o-", linewidth=2, markersize=8)

    # Add labels for each point
    for i, (size, ratio) in enumerate(zip(sizes, cost_ratios)):
        plt.annotate(
            f"{ratio:.2f}",
            (size, ratio),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Set labels and title
    plt.xlabel("Sample Size (n)")
    plt.ylabel("Cost Ratio (Fair / Vanilla)")
    plt.title(title)

    # Use log scale for x-axis if sizes vary significantly
    if max(sizes) / min(sizes) > 5:
        plt.xscale("log", base=2)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Adjust layout and save/show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_parameter_effect(
    df: Any,
    x_param: str,
    y_metric: str,
    hue_param: Optional[str] = None,
    title: str = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the effect of parameters on metrics.

    Args:
        df: DataFrame with experiment results
        x_param: Parameter to plot on x-axis
        y_metric: Metric to plot on y-axis
        hue_param: Parameter to use for color grouping
        title: Plot title
        figsize: Figure size
        save_path: If provided, save the figure to this path
    """
    plt.figure(figsize=figsize)

    # Create plot based on whether we have a hue parameter
    if hue_param:
        sns.lineplot(data=df, x=x_param, y=y_metric, hue=hue_param, marker="o")
    else:
        sns.lineplot(data=df, x=x_param, y=y_metric, marker="o")

    # Set labels and title
    plt.xlabel(x_param)
    plt.ylabel(y_metric)
    if title:
        plt.title(title)
    else:
        plt.title(f"Effect of {x_param} on {y_metric}")

    # Add grid
    plt.grid(True, alpha=0.3)

    # Adjust layout and save/show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_dendrogram_with_scipy(
    points: np.ndarray,
    color_ids: Optional[List[List[int]]] = None,
    title: str = "Hierarchical Clustering Dendrogram",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a dendrogram using scipy's functions directly.
    This avoids conversion issues by using the scipy linkage result directly.

    Args:
        points: The original data points
        color_ids: Lists of point indices for each color category
        title: Plot title
        figsize: Figure size
        save_path: If provided, save the figure to this path
    """
    from scipy.cluster.hierarchy import linkage, dendrogram

    # Compute linkage matrix directly
    Z = linkage(points, method="average")

    # Create figure
    plt.figure(figsize=figsize)

    # Create color mapping for leaf labels if color_ids provided
    leaf_colors = None
    if color_ids is not None:
        n_samples = points.shape[0]

        # Create color map
        color_map = {0: "blue", 1: "red", 2: "green", 3: "purple", 4: "orange"}

        # Assign colors to points
        point_colors = ["gray"] * n_samples
        for color_idx, ids in enumerate(color_ids):
            color = color_map.get(color_idx, f"C{color_idx}")
            for id in ids:
                if 0 <= id < n_samples:
                    point_colors[id] = color

    # Plot dendrogram
    dendrogram(
        Z,
        leaf_rotation=90.0,  # rotates the x axis labels
        leaf_font_size=8.0,  # font size for the x axis labels
        link_color_func=lambda k: "k",  # black links
    )

    # If we have color mappings, add colored dots for the leaves
    if color_ids:
        ax = plt.gca()
        xlbls = ax.get_xticklabels()

        # Extract original indices from labels
        leaf_indices = [int(lbl.get_text()) for lbl in xlbls]

        # Add colored dots
        for i, idx in enumerate(leaf_indices):
            x = float(xlbls[i].get_position()[0])
            color = "gray"
            for color_idx, ids in enumerate(color_ids):
                if idx in ids:
                    color = color_map.get(color_idx, f"C{color_idx}")
                    break
            plt.plot(x, 0, "o", color=color, markersize=8)

        # Add a legend for colors
        legend_elements = []
        for i in range(len(color_ids)):
            from matplotlib.lines import Line2D

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color_map.get(i, f"C{i}"),
                    markersize=10,
                    label=f"Color {i}",
                )
            )
        plt.legend(handles=legend_elements, loc="upper right")

    # Set title and labels
    plt.title(title)
    plt.xlabel("Data Points")
    plt.ylabel("Distance")

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
