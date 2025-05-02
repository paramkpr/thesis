"""
tests/test_visualization_dendrograms.py
Test the dendrogram visualization.
"""

from data.synthetic import generate_colored_data
from algorithms.average_linkage import average_linkage
from algorithms.vanilla import average_linkage as average_linkage_scipy
from visualization.dendograms import plot_colored_tree, plot_dendrogram_with_scipy


def test_dendrogram_visualization():
    # Generate data
    n_points = 50
    points, color_ids = generate_colored_data(
        total_points=n_points, color_proportions=[0.5, 0.3, 0.2], dim=2, seed=42
    )

    # Perform average linkage clustering
    hierarchy = average_linkage(points)

    # Method 1: Plot custom tree visualization
    plot_colored_tree(
        hierarchy=hierarchy,
        color_ids=color_ids,
        title="Custom Tree Visualization",
        figsize=(12, 8),
        save_path="./outs/custom_tree_visualization.png",
    )
    print("Custom tree visualization saved to custom_tree_visualization.png")

    # Method 2: Plot dendrogram directly with scipy
    plot_dendrogram_with_scipy(
        points=points,
        color_ids=color_ids,
        title="SciPy Dendrogram Visualization",
        figsize=(12, 8),
        save_path="./outs/scipy_dendrogram.png",
    )
    print("SciPy dendrogram saved to scipy_dendrogram.png")

    # Method 3: Plot custom tree visualization with scipy linkage
    hierarchy_scipy = average_linkage_scipy(points)
    plot_colored_tree(
        hierarchy=hierarchy_scipy,
        color_ids=color_ids,
        title="Custom Tree Visualization with SciPy Linkage",
        figsize=(12, 8),
        save_path="./outs/custom_tree_visualization_scipy.png",
    )
    print(
        "Custom tree visualization with SciPy linkage saved to custom_tree_visualization_scipy.png"
    )

    plot_dendrogram_with_scipy(
        points=points,
        color_ids=color_ids,
        title="SciPy Dendrogram Visualization with Custom Hierarchy",
        figsize=(12, 8),
        save_path="./outs/scipy_dendrogram_custom_hierarchy.png",
    )
    print(
        "SciPy dendrogram with custom hierarchy saved to scipy_dendrogram_custom_hierarchy.png"
    )


test_dendrogram_visualization()
