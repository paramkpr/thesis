"""
tests/test_algorithms_split_root.py
Test the SplitRoot algorithm.
"""

import math
from data.synthetic import generate_colored_data
from algorithms.vanilla import average_linkage
from algorithms.split_root import split_root
from visualization.dendograms import plot_colored_tree


def test_split_root():
    # Generate data
    n_points = 128
    colors_proportions = [0.5, 0.5]
    points, color_ids = generate_colored_data(
        total_points=n_points, color_proportions=colors_proportions, dim=2, seed=41
    )

    hierarchy = average_linkage(points)

    # Apply SplitRoot
    h = 8  # Desired number of children
    epsilon = 1 / 16  # Balance parameter
    h_prime = split_root(hierarchy, h, epsilon, debug=True)

    # Verify the results
    print(f"Original hierarchy: root has {len(hierarchy.root.children)} children")
    print(f"Modified hierarchy: root has {len(h_prime.root.children)} children")
    print("Root now has", len(h_prime.root.children), "children")
    for ch in h_prime.root.children:
        print("  child", ch.id, "size =", ch.size)

    # Check if balanced
    root_children = h_prime.root.children

    child_sizes = [child.size for child in root_children]
    target_size = n_points / h
    min_target = n_points * (1 / h - epsilon)
    max_target = n_points * (1 / h + epsilon)

    print(f"Target size: {target_size}")
    print(f"Allowed range: [{min_target}, {max_target}]")
    print(f"Actual sizes: {child_sizes}")

    is_balanced = all(min_target <= size <= max_target for size in child_sizes)
    print(f"Is balanced: {is_balanced}")

    # Visualize before and after
    plot_colored_tree(
        hierarchy=hierarchy,
        color_ids=color_ids,
        title="Before SplitRoot",
        figsize=(10, 6),
        save_path="outs/before_split_root.png",
    )

    plot_colored_tree(
        hierarchy=h_prime,
        color_ids=color_ids,
        title="After SplitRoot",
        figsize=(10, 6),
        save_path="outs/after_split_root.png",
    )

    print("Visualizations saved to before_split_root.png and after_split_root.png")


if __name__ == "__main__":
    test_split_root()
