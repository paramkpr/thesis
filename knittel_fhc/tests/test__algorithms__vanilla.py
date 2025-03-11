# Example test code
import numpy as np
from data.synthetic import generate_colored_data
from algorithms.vanilla import average_linkage

# Generate data
n_points = 50
points, color_ids = generate_colored_data(
    total_points=n_points,
    color_proportions=[0.4, 0.6],
    dim=2,
    seed=42
)

# Perform average linkage clustering
hierarchy = average_linkage(points)

# Verify the result
print(f"Number of leaves: {len(hierarchy.get_leaves())}")
print(f"Tree height: {hierarchy.compute_height()}")
print(f"Total nodes: {len(hierarchy.get_all_nodes())}")

# Verify binary property
all_binary = all(len(node.children) <= 2 for node in hierarchy.get_all_nodes())
print(f"All nodes have at most 2 children: {all_binary}")