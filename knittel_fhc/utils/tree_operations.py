"""
utils/tree_operations.py
Author: Param Kapur 
Date: 5/10/25
Sources:
 - Knittel et. al. (https://doi.org/10.48550/arXiv.2311.12501)

Implements cruical Tree Operations:
    - Delete and Insert: (del_ins) which is used in SplitRoot to 
        relocate a subtree in the hierarchy
    - Shallow Fold (shallow_fold): which is used in MakeFair to
        merege a group of sibling subtrees under a new intermediate
        node.
"""
from typing import List, Optional, Dict
from models.node import Node
from models.hierarchy import Hierarchy


def del_ins(hierarchy: Hierarchy, u: Node, v: Node) -> Hierarchy:
    """
    Subtree deletion and insertion operator.
    Delete subtree rooted at u and insert it at v.
    
    Args:
        hierarchy: Hierarchical clustering
        u: Node to delete (non-root)
        v: Node to insert at (not ancestor of u)
        
    Returns:
        Hierarchy: Modified hierarchy
    """
    # Make a copy to avoid modifying the input
    result = hierarchy.copy()
    
    # Find the corresponding nodes in the copied hierarchy
    all_nodes = result.get_all_nodes()
    node_map = {node.id: node for node in all_nodes}
    
    u_copy = node_map.get(u.id)
    v_copy = node_map.get(v.id)
    
    if u_copy is None or v_copy is None:
        raise ValueError(f"Could not find nodes in hierarchy: u_id={u.id}, v_id={v.id}")
    
    # Check if u is the root
    if u_copy.parent is None:
        raise ValueError("Cannot delete the root node")
    
    # Check if v is an ancestor of u
    current = u_copy.parent
    while current is not None:
        if current.id == v_copy.id:
            raise ValueError("Cannot insert at an ancestor of the deleted node")
        current = current.parent
    
    # Remove u from its parent
    u_parent = u_copy.parent
    u_parent.remove_child(u_copy)
    
    # If u's parent now has only one child, contract it
    if len(u_parent.children) == 1:
        sibling = u_parent.children[0]
        grand_parent = u_parent.parent
        
        if grand_parent is not None:
            grand_parent.remove_child(u_parent)
            grand_parent.add_child(sibling)
        else:
            # u_parent was the root
            result.root = sibling
            sibling.parent = None
    
    # Insert u at v
    v_parent = v_copy.parent
    if v_parent is None:
        # v is the root, create new root
        max_id = max(node.id for node in all_nodes) if all_nodes else 0
        new_root = Node(
            node_id=max_id + 1,
            children=[],
            color_counts={}  # Initialize with empty dict
        )
        result.root = new_root
        new_root.add_child(v_copy)
        new_root.add_child(u_copy)
    else:
        # Create new parent for v and u
        max_id = max(node.id for node in all_nodes) if all_nodes else 0
        new_parent = Node(
            node_id=max_id + 1,
            children=[],
            color_counts={}  # Initialize with empty dict
        )
        v_parent.remove_child(v_copy)
        v_parent.add_child(new_parent)
        new_parent.add_child(v_copy)
        new_parent.add_child(u_copy)
    
    # Update sizes
    result.update_all_sizes()
    
    return result