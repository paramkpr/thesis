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
import logging
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


def del_ins(hierarchy: Hierarchy, u: Node, v: Node) -> Hierarchy:
    """
    Enhanced deletion and insertion operator with validation.
    """
    # Make a copy
    result = hierarchy.copy()
    
    # Map nodes to the copied hierarchy
    all_nodes = result.get_all_nodes()
    node_map = {node.id: node for node in all_nodes}
    
    u_copy = node_map.get(u.id)
    v_copy = node_map.get(v.id)
    
    if u_copy is None or v_copy is None:
        print(f"Warning: Could not find nodes u_id={u.id} or v_id={v.id}")
        return result
    
    # Validation checks
    if u_copy.parent is None:
        print(f"Warning: Cannot delete the root node (id={u_copy.id})")
        return result
        
    # Check for ancestry relationship
    is_ancestor = False
    current = v_copy.parent
    while current:
        if current.id == u_copy.id:
            is_ancestor = True
            break
        current = current.parent
        
    if is_ancestor:
        print(f"Warning: Cannot insert at descendant node (u_id={u_copy.id}, v_id={v_copy.id})")
        return result
    
    # Store references
    u_parent = u_copy.parent
    
    # Remove u from its parent
    u_parent.remove_child(u_copy)
    
    # Handle parent contraction if needed
    if len(u_parent.children) == 1:
        sibling = u_parent.children[0]
        grand_parent = u_parent.parent
        
        if grand_parent:
            grand_parent.remove_child(u_parent)
            grand_parent.add_child(sibling)
        else:
            # u_parent was root
            result.root = sibling
            sibling.parent = None
    
    # Insert u at v
    v_parent = v_copy.parent
    if v_parent is None:
        # v is root
        max_id = max(node.id for node in all_nodes) + 1
        new_root = Node(node_id=max_id, children=[], color_counts={})
        result.root = new_root
        new_root.add_child(v_copy)
        new_root.add_child(u_copy)
    else:
        # Create new parent for v and u
        max_id = max(node.id for node in all_nodes) + 1
        new_parent = Node(node_id=max_id, children=[], color_counts={})
        v_parent.remove_child(v_copy)
        v_parent.add_child(new_parent)
        new_parent.parent = v_parent
        new_parent.add_child(v_copy)
        new_parent.add_child(u_copy)
    
    # Update sizes
    result.update_all_sizes()
    
    return result

def shallow_fold(hierarchy: Hierarchy, nodes_to_fold: List[Node]) -> Hierarchy:
    """
    Shallow tree folding operator with additional debugging.
    """
    # Make a copy to avoid modifying the input
    result = hierarchy.copy()
    
    # Find the corresponding nodes in the copied hierarchy
    all_nodes = result.get_all_nodes()
    node_map = {node.id: node for node in all_nodes}
    
    # Map the input nodes to their copies in the new hierarchy
    nodes_to_fold_copies = [node_map.get(node.id) for node in nodes_to_fold if node.id in node_map]
    
    # Ensure there are nodes to fold
    if not nodes_to_fold_copies:
        logger.warning("No nodes to fold after mapping!")
        return result
    
    # Log the nodes we're folding
    logger.debug(f"Folding nodes: {[n.id for n in nodes_to_fold_copies]}")
    
    # Ensure all nodes have the same parent
    parent_ids = {node.parent.id if node.parent else None for node in nodes_to_fold_copies}
    if len(parent_ids) != 1:
        logger.error(f"Nodes to fold have different parents: {parent_ids}")
        return result  # Return unchanged instead of raising an exception
    
    # Get the parent
    parent = nodes_to_fold_copies[0].parent
    if parent is None:
        logger.error("Cannot fold root nodes!")
        return result  # Return unchanged
    
    # Create a new node to hold the folded subtrees
    max_id = max(node.id for node in all_nodes) if all_nodes else 0
    folded_node = Node(
        node_id=max_id + 1,
        parent=None,
        children=[],
        data_indices=[],
        size=0,
        color_counts={}
    )
    
    # Collect all children from nodes to fold
    all_children = []
    all_data_indices = []
    
    try:
        for node in nodes_to_fold_copies:
            # Log what we're removing
            logger.debug(f"Removing node {node.id} from parent {parent.id}")
            
            parent.remove_child(node)
            all_children.extend(node.children)
            all_data_indices.extend(node.data_indices)
            
            # Update parent references for the children
            for child in node.children:
                child.parent = None
    except Exception as e:
        logger.error(f"Error during fold operation: {e}")
        # Continue with what we've got so far
    
    # Add all children to the folded node
    for child in all_children:
        folded_node.add_child(child)
    
    # Set the data indices for the folded node
    folded_node.data_indices = all_data_indices
    
    # Add the folded node to the parent
    logger.debug(f"Adding folded node {folded_node.id} to parent {parent.id}")
    parent.add_child(folded_node)
    
    # Update sizes
    result.update_all_sizes()
    
    return result