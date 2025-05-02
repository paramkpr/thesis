"""
data/synthetic.py
Author: Param Kapur
Date: 5/10/25
Sources:
 - Knittel et. al. (https://doi.org/10.48550/arXiv.2311.12501)


Generates points from gaussian clusters.
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt


def generate_colored_data(
    total_points: int,
    color_proportions: List[float],
    dim: int = 2,
    cluster_std: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Generate data points with random color assignments.
    Colors are assigned independently of spatial location.

    Args:
        total_points: Total number of points to generate
        color_proportions: List of proportions for each color (should sum to 1.0)
        dim: Dimensionality of points
        cluster_std: Standard deviation of clusters
        seed: Random seed

    Returns:
        points: Array of data points
        color_ids: List of lists containing point indices for each color, a list of list of the ids of the points of each color
    """
    if seed is not None:
        np.random.seed(seed)

    # Validate proportions
    if abs(sum(color_proportions) - 1.0) > 1e-10:
        raise ValueError(
            f"Color proportions must sum to 1.0, got {sum(color_proportions)}"
        )

    # First, generate all points without color assignment
    # Let's create some natural clusters (e.g., 4-5 spatial clusters)
    num_spatial_clusters = 4
    # Generate cluster centers
    spatial_centers = np.random.uniform(-10, 10, size=(num_spatial_clusters, dim))

    # Assign each point to a random spatial cluster
    cluster_sizes = np.random.multinomial(
        total_points, [1 / num_spatial_clusters] * num_spatial_clusters
    )

    # Generate all points
    points = []
    for i in range(num_spatial_clusters):
        cluster_points = np.random.normal(
            loc=spatial_centers[i], scale=cluster_std, size=(cluster_sizes[i], dim)
        )
        points.extend(cluster_points)

    points = np.array(points)

    # Now assign colors completely independently of spatial location
    # Calculate the number of points for each color
    num_colors = len(color_proportions)
    color_counts = []
    remaining_points = total_points

    for i, prop in enumerate(color_proportions[:-1]):
        count = int(total_points * prop)
        color_counts.append(count)
        remaining_points -= count

    # Assign remaining points to the last color
    color_counts.append(remaining_points)

    # Randomly assign colors by randomly permuting point indices
    all_indices = np.arange(total_points)
    np.random.shuffle(all_indices)  # This is the key - we shuffle all indices

    # Split indices by color
    color_ids = []
    start_idx = 0

    for count in color_counts:
        color_indices = all_indices[start_idx : start_idx + count].tolist()
        color_ids.append(color_indices)
        start_idx += count

    return points, color_ids
