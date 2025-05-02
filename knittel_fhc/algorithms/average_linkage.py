# ── algorithms/average_linkage.py ────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from typing import List, Tuple

from models.node import Node
from models.hierarchy import Hierarchy


def _pairwise_euclidean(points: np.ndarray) -> np.ndarray:
    """
    Compute the full n×n Euclidean distance matrix in O(n² d).
    """
    # Broadcasting trick: (x‑y)² summed over last axis
    diffs = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diffs**2).sum(axis=2))
    return dist


def _distance_to_similarity(dist: np.ndarray) -> np.ndarray:
    """
    Transform distances → similarities via  s_ij = 1 / (1 + d_ij).
    """
    with np.errstate(divide="ignore"):
        sim = 1.0 / (1.0 + dist)
    np.fill_diagonal(sim, 0.0)  # self‑similarity not used
    return sim


def _find_argmax(mat: np.ndarray) -> Tuple[int, int]:
    """
    Return (i, j) with i < j such that mat[i, j] is maximal.
    """
    n = mat.shape[0]
    # Upper‑triangular (excluding diag) linear index of max element
    iu = np.triu_indices(n, k=1)
    flat_idx = np.argmax(mat[iu])
    return int(iu[0][flat_idx]), int(iu[1][flat_idx])


def average_linkage(points: np.ndarray) -> Hierarchy:
    """
    Standard average‑linkage agglomerative clustering (MAX similarity rule).

    Args:
        points (np.ndarray): shape (n, d) raw data.

    Returns:
        Hierarchy: root node of the constructed dendrogram.
    """
    n = points.shape[0]
    if n == 0:
        return Hierarchy(root=None)

    # 1. Pre‑compute similarity matrix
    dist = _pairwise_euclidean(points)
    sim = _distance_to_similarity(dist)

    # 2. Initialise leaves
    clusters: List[Node] = [Node(node_id=i, data_indices=[i], size=1) for i in range(n)]
    next_id = n  # IDs for newly created internal nodes follow the leaves

    # 3. Agglomerative loop
    while len(clusters) > 1:
        # --- 3.1 choose pair with maximum similarity
        i, j = _find_argmax(sim)
        if i > j:  # keep i < j for convenience
            i, j = j, i

        left = clusters[i]
        right = clusters[j]

        # --- 3.2 create new parent node
        merged = Node(
            node_id=next_id,
            children=[left, right],
            data_indices=left.data_indices + right.data_indices,
            size=left.size + right.size,
        )
        left.parent = merged
        right.parent = merged
        next_id += 1

        # --- 3.3 update data structures
        #   • append merged cluster
        #   • update similarity matrix by average linkage rule
        left_w, right_w = left.size, right.size
        new_row = (sim[i] * left_w + sim[j] * right_w) / (left_w + right_w)
        new_row = np.append(new_row, 0.0)  # self‑similarity placeholder

        # insert new_row as last row & col BEFORE deleting i,j
        sim = np.vstack((sim, new_row[:-1]))
        sim = np.hstack((sim, new_row.reshape(-1, 1)))

        # remove rows/cols of merged children (delete higher index first)
        for idx in sorted([i, j], reverse=True):
            sim = np.delete(sim, idx, axis=0)
            sim = np.delete(sim, idx, axis=1)
            clusters.pop(idx)

        # add merged node
        clusters.append(merged)

    # 4. Wrap in Hierarchy and return
    root = clusters[0]
    hierarchy = Hierarchy(root=root)
    hierarchy.update_all_sizes()  # sizes already OK, but keep it clean
    return hierarchy
