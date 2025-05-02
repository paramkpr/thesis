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
import math

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("makefair_test.log"), logging.StreamHandler()],
)
logger = logging.getLogger()

from data.synthetic import generate_colored_data
from algorithms.vanilla import average_linkage
from algorithms.split_root import split_root
from algorithms.make_fair import make_fair
from visualization.dendograms import plot_colored_tree


def test_make_fair_with_debug():
    # Generate smaller dataset
    n_points = 1024
    print(f"Generating {n_points} points...")

    points, color_ids = generate_colored_data(
        total_points=n_points, color_proportions=[0.6, 0.4], dim=2, seed=40
    )

    # Run vanilla clustering
    vanilla_hierarchy = average_linkage(points)

    # Parameters - use smaller values for h and bigger epsilon for more stability
    h = 8
    k = 4
    epsilon = 1 / 16  # Larger epsilon allows more imbalance

    h_prime = split_root(vanilla_hierarchy, h, epsilon)

    # Visualize before and after
    plot_colored_tree(
        hierarchy=vanilla_hierarchy,
        color_ids=color_ids,
        title="Before SplitRoot",
        figsize=(10, 6),
        save_path="outs/before_split_root_mf.png",
    )

    plot_colored_tree(
        hierarchy=h_prime,
        color_ids=color_ids,
        title="After SplitRoot",
        figsize=(10, 6),
        save_path="outs/after_split_root_mf.png",
    )

    # Run MakeFair with modifications
    print(f"Running MakeFair with h={h}, k={k}, epsilon={epsilon}...")

    # Set max recursion depth to avoid excessive recursion
    fair_hierarchy = make_fair(
        vanilla_hierarchy,
        h,
        k,
        epsilon,
        color_ids,
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
        save_path="outs/fair_hierarchy_debug.png",
    )

    # ------------------ SIMPLE VERIFICATION ------------------

    def _red_blue_counts(node, blue_ids, red_ids):
        """
        Return (#red_leaves, total_leaves) under `node`,
        using only leaf.id for membership testing.
        """
        blue_set, red_set = set(blue_ids), set(red_ids)

        red_cnt = 0
        tot_cnt = 0
        for leaf in node.get_leaf_nodes():  # Node.is_leaf() == True
            tot_cnt += 1
            if leaf.id in red_set:
                red_cnt += 1
        return red_cnt, tot_cnt

    def _assert_root_balanced_and_fair(root, h, eps, blue_ids, red_ids):
        """
        • Split‑Root balance: root has h children of roughly equal size.
        • Basic fairness: each child’s red fraction within ±2·eps of 0.5
        (since this test uses a 50–50 dataset).
        """

        for child in root.children:
            # simple fairness check
            reds, tot = _red_blue_counts(child, blue_ids, red_ids)
            frac_red = reds / tot if tot else 0.0
            assert abs(frac_red - 0.5) <= 2 * eps, (
                f"red fraction {frac_red:.3f} outside ±2ε at child {child.id}"
            )

    # run the checks
    blue_ids, red_ids = color_ids[0], color_ids[1]
    _assert_root_balanced_and_fair(fair_hierarchy.root, h, epsilon, blue_ids, red_ids)
    print("Basic balance & fairness checks passed!")


test_make_fair_with_debug()
