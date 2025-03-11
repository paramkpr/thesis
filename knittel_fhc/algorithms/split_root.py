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


def split_root(hierarchy: Hierarchy, h: int, epsilon: float, max_iterations: int = 100) -> Hierarchy:
    """
    Implements the SplitRoot algorithm with safety limits.
    
    Args:
        hierarchy: The input hierarchy
        h: The desired number of children
        epsilon: The balance parameter
        max_iterations: Maximum number of balancing iterations to prevent infinite loops
        
    Returns:
        Hierarchy: Modified hierarchy with balanced root
    """
    # Create a copy
    result = hierarchy.copy()
    result.update_all_sizes()
    root = result.root
    
    # Ensure h children
    current_children = root.children.copy()
    next_id = max(node.id for node in result.get_all_nodes()) + 1 if result.get_all_nodes() else 0
    
    while len(root.children) < h:
        dummy = Node(
            node_id=next_id,
            parent=None,
            data_indices=[],
            size=0,
            color_counts={}
        )
        next_id += 1
        root.add_child(dummy)
    
    # Total leaves
    total_leaves = root.size
    
    # Balancing with iteration limit
    iteration_count = 0
    
    while iteration_count < max_iterations:
        iteration_count += 1
        
        # Find min and max children
        children = root.children
        if not children:
            break
            
        vmin = min(children, key=lambda c: c.size)
        vmax = max(children, key=lambda c: c.size)
        
        # Check if balanced
        min_target = total_leaves * (1/h - epsilon)
        max_target = total_leaves * (1/h + epsilon)
        
        if vmax.size <= max_target and vmin.size >= min_target:
            break  # Balanced
        
        # Calculate shift amount
        delta1 = (1/h) - (vmin.size / total_leaves)
        delta2 = (vmax.size / total_leaves) - (1/h)
        delta = min(delta1, delta2)
        delta_nodes = int(delta * total_leaves)
        
        # Safety check: ensure we're making progress
        if delta_nodes < 1:
            # If we can't move at least one node, force a minimal move
            delta_nodes = 1
        
        # Find subtree to move
        v = vmax
        while not v.is_leaf() and v.size > delta_nodes:
            # Find largest child
            largest_child = max(v.children, key=lambda c: c.size) if v.children else None
            if largest_child is None:
                break
            v = largest_child
        
        # Find insertion point
        u = vmin
        while not u.is_leaf():
            if not u.children:
                break
                
            smallest_child = min(u.children, key=lambda c: c.size)
            
            if smallest_child.size < v.size:
                u = smallest_child
            else:
                break
        
        # Safety check: don't move a node to itself or a descendant
        current = u
        is_descendant = False
        while current:
            if current.id == v.id:
                is_descendant = True
                break
            current = current.parent
            
        if is_descendant:
            print(f"Warning: Attempted to move node {v.id} to its descendant {u.id}")
            break
        
        # Perform the move
        result = del_ins(result, v, u)
        root = result.root  # Update root reference
    
    if iteration_count >= max_iterations:
        print(f"Warning: SplitRoot reached max iterations ({max_iterations})")
    
    return result