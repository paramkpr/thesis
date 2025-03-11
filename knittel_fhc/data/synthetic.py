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
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Generate data points with colors.
    
    Args:
        total_points: Total number of points to generate
        color_proportions: List of proportions for each color (should sum to 1.0)
        dim: Dimensionality of points
        cluster_std: Standard deviation of clusters
        seed: Random seed
        
    Returns:
        points: Array of data points
        color_ids: List of lists containing point indices for each color
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    # Validate proportions
    if abs(sum(color_proportions) - 1.0) > 1e-10:
        raise ValueError(f"Color proportions must sum to 1.0, got {sum(color_proportions)}")
    
    # Calculate counts for each color
    num_colors = len(color_proportions)
    color_counts = []
    remaining_points = total_points
    
    for i, prop in enumerate(color_proportions[:-1]):
        count = int(total_points * prop)
        color_counts.append(count)
        remaining_points -= count
    
    # Assign remaining points to the last color
    color_counts.append(remaining_points)
    
    # Generate points with different distributions for each color
    points = []
    color_ids = [[] for _ in range(num_colors)]
    current_idx = 0
    
    # Generate centers that are reasonably separated
    centers = np.random.uniform(-10, 10, size=(num_colors, dim))
    
    # For each color, generate points around a center
    for color_idx, count in enumerate(color_counts):
        # Generate points for this color around its center
        color_points = np.random.normal(
            loc=centers[color_idx], 
            scale=cluster_std, 
            size=(count, dim)
        )
        
        # Add points to the list
        points.extend(color_points)
        
        # Record indices for this color
        color_ids[color_idx] = list(range(current_idx, current_idx + count))
        current_idx += count
    
    return np.array(points), color_ids