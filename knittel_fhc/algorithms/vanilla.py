"""
algorithms/vanilla.py
Author: Param Kapur 
Date: 5/10/25
Sources:
 - Knittel et. al. (https://doi.org/10.48550/arXiv.2311.12501)

Implements vanilla (unfair) average linkage hierarchical clustering.
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy.cluster.hierarchy import linkage, to_tree
from models.node import Node
from models.hierarchy import Hierarchy


def average_linkage(points: np.ndarray) -> Hierarchy:
    """
    Perform average-linkage hierarchical clustering using SciPy.
    
    Args:
        points: 2D array of observation vectors with shape (n_samples, n_features)
        
    Returns:
        Hierarchy object representing the clustering
    """
    Z = linkage(points, method='average', metric='euclidean')
    
    hierarchy = scipy_linkage_to_hierarchy(Z, n_samples=points.shape[0])
    hierarchy.update_all_sizes()
    
    return hierarchy


def scipy_linkage_to_hierarchy(Z: np.ndarray, n_samples: int) -> Hierarchy:
    """
    Convert a scipy linkage matrix to our custom Hierarchy.
    """
    # Convert to scipy tree representation
    scipy_tree = to_tree(Z, rd=True)
    scipy_root = scipy_tree[0]
    
    # Start building from scipy's representation
    # We use n_samples as the starting ID for internal nodes (scipy's convention)
    next_id = n_samples
    custom_root = scipy_node_to_custom_node(scipy_root, parent=None, 
                                           next_id=next_id, n_samples=n_samples)
    
    return Hierarchy(root=custom_root)


def scipy_node_to_custom_node(scipy_node, parent: Optional[Node], 
                             next_id: int, n_samples: int) -> Node:
    """
    Convert a scipy ClusterNode to our custom Node.
    
    Args:
        scipy_node: scipy ClusterNode object
        parent: Parent node (None for root)
        next_id: ID to use for non-leaf nodes if scipy doesn't provide one
        n_samples: Number of original data points
        
    Returns:
        Node: Custom node object
    """
    if scipy_node.is_leaf():
        # Scipy leaf nodes have IDs in range [0, n_samples-1]
        node_id = int(scipy_node.id)
        
        node = Node(
            node_id=node_id,
            parent=parent,
            data_indices=[node_id],  # Leaf node represents a single data point
            size=1,
        )
        return node
    
    # For internal nodes
    # Scipy assigns IDs starting from n_samples to internal nodes
    node_id = int(scipy_node.id) if hasattr(scipy_node, 'id') else next_id
    next_id = max(next_id, node_id + 1)
    
    # Create internal node
    node = Node(
        node_id=node_id,
        parent=parent,
        size=int(scipy_node.count),  # scipy provides count information
    )
    
    # Recursively convert children
    left_child = scipy_node_to_custom_node(scipy_node.left, parent=node, 
                                          next_id=next_id, n_samples=n_samples)
    right_child = scipy_node_to_custom_node(scipy_node.right, parent=node, 
                                           next_id=next_id, n_samples=n_samples)
    
    # Add children to node
    node.add_child(left_child)
    node.add_child(right_child)
    
    # Combine data_indices from children for internal nodes
    node.data_indices = left_child.data_indices + right_child.data_indices
    
    return node