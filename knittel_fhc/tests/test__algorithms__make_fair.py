"""
tests/test_algorithms_make_fair_debug.py
Enhanced test for the MakeFair algorithm with debugging.
"""

import sys
import os
import time
import logging
import matplotlib.pyplot as plt
import numpy as np

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("makefair_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

from data.synthetic import generate_colored_data
from algorithms.vanilla import average_linkage
from algorithms.split_root import split_root
from algorithms.make_fair import make_fair
from visualization.dendograms import plot_colored_tree

def test_make_fair_with_debug():
    # Generate smaller dataset
    n_points = 128
    print(f"Generating {n_points} points...")
    
    points, color_ids = generate_colored_data(
        total_points=n_points,
        color_proportions=[0.4, 0.3, 0.3],
        dim=2,
        seed=42
    )
    
    # Run vanilla clustering
    vanilla_hierarchy = average_linkage(points)
    
    # Parameters - use smaller values for h and bigger epsilon for more stability
    h = 8
    k = 4
    epsilon = 1/6  # Larger epsilon allows more imbalance
    
    h_prime = split_root(vanilla_hierarchy, h, epsilon)

    # Visualize before and after
    plot_colored_tree(
        hierarchy=vanilla_hierarchy,
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

    # Run MakeFair with modifications
    print(f"Running MakeFair with h={h}, k={k}, epsilon={epsilon}...")
    
    # Set max recursion depth to avoid excessive recursion
    fair_hierarchy = make_fair(
        vanilla_hierarchy, h, k, epsilon, color_ids, 
        max_depth=4,  # Limit recursion depth
        debug=True    # Extensive logging
    )
    
    # Verify result
    print("MakeFair completed successfully!")
    print(f"Original hierarchy had {len(vanilla_hierarchy.get_leaves())} leaves")
    print(f"Fair hierarchy has {len(fair_hierarchy.get_leaves())} leaves")
    
    # Visualize results
    plot_colored_tree(
        hierarchy=fair_hierarchy,
        color_ids=color_ids,
        title="Fair Hierarchical Clustering",
        save_path="outs/fair_hierarchy_debug.png"
    )

test_make_fair_with_debug()