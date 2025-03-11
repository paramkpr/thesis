"""
algorithms/split_root.py
Author: Param Kapur 
Date: 5/10/25
Sources:
 - Knittel et. al. (https://doi.org/10.48550/arXiv.2311.12501)

Implements the SplitRoot algorithm from the paper, which ensures
that the root of a hierarchy has h children of roughly equal size.
"""

from typing import List, Optional, Dict
from models.node import Node
from models.hierarchy import Hierarchy
from utils.tree_operations import del_ins


def split_root(hierarchy: Hierarchy, h: int, epsilon: float) -> Hierarchy:
    """
    Implements the SplitRoot algorithm to balance the root of a hierarchical clustering.
    
    Ensures the root node has `h` children that are roughly balanced in size.

    Args:
        hierarchy: The input hierarchy.
        h: The desired number of children at the root.
        epsilon: The balance parameter (0 < ε < 1/6).

    Returns:
        Hierarchy: A modified hierarchy with a balanced root.
    """
    # Create a copy to avoid modifying the original hierarchy
    result = hierarchy.copy()
    result.update_all_sizes()
    
    root = result.root
    
    all_nodes = result.get_all_nodes()
    next_id = max(node.id for node in all_nodes) + 1 if all_nodes else 0
    
    # Add dummy nodes if needed
    while len(root.children) < h:
        dummy = Node(
            node_id=next_id,
            parent=None,
            data_indices=[],
            size=0,
            color_counts={}  # Initialize with empty dict
        )
        next_id += 1
        root.add_child(dummy)
    
    total_leaves = root.size
    
    # Main loop: repeatedly balance the root
    while True:
        # Find the smallest and largest child clusters
        children = root.children
        if not children:
            break
        
        vmin = min(children, key=lambda c: c.size)
        vmax = max(children, key=lambda c: c.size)
        
        # Check if the root is already balanced
        min_target = total_leaves * (1/h - epsilon)
        max_target = total_leaves * (1/h + epsilon)
        
        if vmax.size <= max_target and vmin.size >= min_target:
            break  # If balanced, stop the process
        
        # Calculate how much needs to be shifted
        delta1 = (1/h) - (vmin.size / total_leaves)  # How much vmin is below target
        delta2 = (vmax.size / total_leaves) - (1/h)  # How much vmax is above target
        delta = min(delta1, delta2)
        
        # Find the subtree in `vmax` to move
        v = vmax
        while not v.is_leaf() and v.size > delta * total_leaves:
            largest_child = max(v.children, key=lambda c: c.size)
            v = largest_child
        
        # Find the insertion point in `vmin`
        u = vmin
        while not u.is_leaf():
            # Find the smallest child
            children = u.children
            if not children:
                break
                
            smallest_child = min(children, key=lambda c: c.size)
            
            # If the right child of the current node is smaller than v,
            # move down to the left child
            if smallest_child.size < v.size:
                u = smallest_child
            else:
                break
        
        # Move `v` under `u` using del_ins operation
        result = del_ins(result, v, u)
        
        # After del_ins, we need to update the root reference
        # since the hierarchy might have been restructured
        root = result.root
    
    return result