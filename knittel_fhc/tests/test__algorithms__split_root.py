"""
tests/test_algorithms_split_root.py
Test the SplitRoot algorithm.
"""
from data.synthetic import generate_colored_data
from algorithms.vanilla import average_linkage
from algorithms.split_root import split_root
from visualization.dendograms import plot_colored_tree

def test_split_root():
    # Generate data
    n_points = 128
    points, color_ids = generate_colored_data(
        total_points=n_points,
        color_proportions=[0.4, 0.3, 0.2, 0.1],
        dim=2,
        seed=42
    )
    
    hierarchy = average_linkage(points)
    
    # Apply SplitRoot
    h = 4  # Desired number of children
    epsilon = 1/16  # Balance parameter
    h_prime = split_root(hierarchy, h, epsilon)
    
    # Verify the results
    print(f"Original hierarchy: root has {len(hierarchy.root.children)} children")
    print(f"Modified hierarchy: root has {len(h_prime.root.children)} children")
    
    # Check if balanced
    root_children = h_prime.root.children
    total_leaves = h_prime.root.size
    
    child_sizes = [child.size for child in root_children]
    target_size = total_leaves / h
    min_target = total_leaves * (1/h - epsilon)
    max_target = total_leaves * (1/h + epsilon)
    
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
        save_path="outs/before_split_root.png"
    )
    
    plot_colored_tree(
        hierarchy=h_prime,
        color_ids=color_ids,
        title="After SplitRoot",
        figsize=(10, 6),
        save_path="outs/after_split_root.png"
    )

    print("Visualizations saved to before_split_root.png and after_split_root.png")

if __name__ == "__main__":
    test_split_root()