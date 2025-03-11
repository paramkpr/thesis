"""
utils/distance.py
Author: Param Kapur 
Date: 5/10/25
Sources:
 - Knittel et. al. (https://doi.org/10.48550/arXiv.2311.12501)


Implements distance and similarity computations for clustering.
"""

from __future__ import annotations
from typing import List, Callable
import math
import numpy as np


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """
    Compute the Euclidean distance between two vectors a and b.
    Args:
        a (List[float]): First vector.
        b (List[float]): Second vector.
    Returns:
        float: Euclidean distance.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must be the same dimension for Euclidean distance.")
    return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))

def manhattan_distance(a: List[float], b: List[float]) -> float:
    """
    Compute the Manhattan (L1) distance between two vectors a and b.
    Args:
        a (List[float]): First vector.
        b (List[float]): Second vector.
    Returns:
        float: Manhattan distance.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must be the same dimension for Manhattan distance.")
    return sum(abs(ai - bi) for ai, bi in zip(a, b))

def convert_distance_to_similarity(d: float) -> float:
    """
    Convert a distance value to a similarity score using 1 / (1 + d).
    Args:
        d (float): Distance.
    Returns:
        float: Similarity score in (0, 1].
    """
    return 1.0 / (1.0 + d)


def calculate_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance matrix between points.
    
    Args:
        points: Array of shape (n_samples, n_features)
        
    Returns:
        distance_matrix: Square matrix of pairwise distances
    """
    n = len(points)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):  # Only calculate upper triangle
            # Euclidean distance
            dist = euclidean_distance(points[i], points[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Distance matrix is symmetric
    
    return distance_matrix


def convert_to_similarity(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Convert distance matrix to similarity matrix using 1/(1+d).
    
    Args:
        distance_matrix: Square matrix of pairwise distances
        
    Returns:
        similarity_matrix: Square matrix of pairwise similarities
    """
    similarity_matrix = 1.0 / (1.0 + distance_matrix)
    np.fill_diagonal(similarity_matrix, 1.0)  # Self-similarity = 1
    
    return similarity_matrix