"""
tests/test_visualization_dendrograms.py
Test the dendrogram visualization.
"""
from data.synthetic import generate_colored_data
from algorithms.vanilla import average_linkage
from visualization.dendograms import plot_colored_tree, plot_dendrogram_with_scipy

def test_dendrogram_visualization():
    # Generate data
    n_points = 50
    points, color_ids = generate_colored_data(
        total_points=n_points,
        color_proportions=[0.4, 0.3, 0.3],
        dim=2,
        seed=42
    )
    
    # Perform average linkage clustering
    hierarchy = average_linkage(points)
    
    # Method 1: Plot custom tree visualization
    plot_colored_tree(
        hierarchy=hierarchy,
        color_ids=color_ids,
        title="Custom Tree Visualization",
        figsize=(12, 8),
        save_path="./outs/custom_tree_visualization.png"
    )
    print("Custom tree visualization saved to custom_tree_visualization.png")
    
    # Method 2: Plot dendrogram directly with scipy
    plot_dendrogram_with_scipy(
        points=points,
        color_ids=color_ids,
        title="SciPy Dendrogram Visualization",
        figsize=(12, 8),
        save_path="./outs/scipy_dendrogram.png"
    )
    print("SciPy dendrogram saved to scipy_dendrogram.png")

test_dendrogram_visualization()