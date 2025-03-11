"""
algorithms/make_fair.py
Author: Param Kapur 
Date: 5/10/25
Sources:
 - Knittel et. al. (https://doi.org/10.48550/arXiv.2311.12501)

Implements the MakeFair algorithm with debugging capabilities.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from models.node import Node
from models.hierarchy import Hierarchy
from algorithms.split_root import split_root
from utils.tree_operations import shallow_fold

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("makefair_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MakeFair")

# Debug stats to track progress
debug_stats = {
    "recursion_depth": 0,
    "max_recursion_depth": 0,
    "nodes_processed": 0,
    "folds_performed": 0,
    "start_time": 0,
    "step_times": {},
}

def reset_debug_stats():
    """Reset debugging statistics"""
    global debug_stats
    debug_stats = {
        "recursion_depth": 0,
        "max_recursion_depth": 0,
        "nodes_processed": 0,
        "folds_performed": 0,
        "start_time": time.time(),
        "step_times": {},
    }

def log_step_time(step_name, start_time):
    """Log the time taken for a step"""
    elapsed = time.time() - start_time
    if step_name in debug_stats["step_times"]:
        debug_stats["step_times"][step_name] += elapsed
    else:
        debug_stats["step_times"][step_name] = elapsed
    return elapsed

def make_fair(
    hierarchy: Hierarchy, 
    h: int, 
    k: int, 
    epsilon: float,
    color_ids: List[List[int]],
    max_depth: int = 10,  # Add max recursion depth parameter
    debug: bool = True
) -> Hierarchy:
    """
    Implements the MakeFair algorithm to create a fair hierarchical clustering.
    
    Args:
        hierarchy: Input hierarchical clustering
        h: Parameter for SplitRoot (desired number of children)
        k: Parameter for folding (number of groups to partition children into)
        epsilon: Balance parameter (0 < ε < 1/6)
        color_ids: Lists of point indices for each color category
        max_depth: Maximum recursion depth (for safety)
        debug: Whether to print debug information
        
    Returns:
        Hierarchy: Fair hierarchical clustering
    """
    # Initialize debug stats on first call
    if debug_stats["start_time"] == 0:
        reset_debug_stats()
    
    # Increment recursion depth
    debug_stats["recursion_depth"] += 1
    debug_stats["max_recursion_depth"] = max(
        debug_stats["max_recursion_depth"], 
        debug_stats["recursion_depth"]
    )
    current_depth = debug_stats["recursion_depth"]
    
    # Log start of this recursion level
    if debug:
        logger.info(f"[Depth {current_depth}] Starting MakeFair on hierarchy with {len(hierarchy.get_leaves())} leaves")
        logger.info(f"[Depth {current_depth}] Parameters: h={h}, k={k}, epsilon={epsilon}")
        
        # Print node sizes at start
        if hierarchy.root:
            logger.info(f"[Depth {current_depth}] Root size: {hierarchy.root.size}")
            child_sizes = [child.size for child in hierarchy.root.children] if hierarchy.root.children else []
            logger.info(f"[Depth {current_depth}] Child sizes: {child_sizes}")
    
    # Safety check - too deep recursion
    if current_depth > max_depth:
        if debug:
            logger.warning(f"[Depth {current_depth}] Exceeded max recursion depth of {max_depth}!")
        debug_stats["recursion_depth"] -= 1
        return hierarchy
    
    # Make a copy to avoid modifying the input
    result = hierarchy.copy()
    
    # Step 1: Apply SplitRoot to ensure the root has h children
    step_start = time.time()
    if debug:
        logger.info(f"[Depth {current_depth}] Applying SplitRoot...")
    
    result = split_root(result, h, epsilon)
    
    if debug:
        elapsed = log_step_time("split_root", step_start)
        logger.info(f"[Depth {current_depth}] SplitRoot complete in {elapsed:.2f}s. Root now has {len(result.root.children)} children.")
    
    # Step 2: Assign colors to nodes based on color_ids
    step_start = time.time()
    if debug:
        logger.info(f"[Depth {current_depth}] Assigning colors...")
    
    _assign_colors(result, color_ids)
    
    if debug:
        elapsed = log_step_time("assign_colors", step_start)
        logger.info(f"[Depth {current_depth}] Color assignment complete in {elapsed:.2f}s")
    
    # Save the original number of children
    h_prime = len(result.root.children)
    
    # Step 3: For each color, apply folding
    step_start = time.time()
    num_colors = len(color_ids)
    
    if debug:
        logger.info(f"[Depth {current_depth}] Starting folding for {num_colors} colors with k={k}...")
    
    for color_idx in range(num_colors):
        if debug:
            logger.info(f"[Depth {current_depth}] Processing color {color_idx}...")
        
        # Sort children by the proportion of this color (decreasing)
        _sort_children_by_color(result.root, color_idx)
        
        # Get updated list of children (might have changed after previous color)
        children = result.root.children.copy()
        child_sizes = [child.size for child in children]
        
        if debug:
            logger.info(f"[Depth {current_depth}] Child sizes after sorting for color {color_idx}: {child_sizes}")
        
        # Partition children into k groups
        child_groups = _partition_into_groups(children, k)
        
        if debug:
            group_sizes = [len(group) for group in child_groups]
            logger.info(f"[Depth {current_depth}] Partitioned into {len(child_groups)} groups with sizes {group_sizes}")
        
        # Apply shallow folding to each group
        for i, group in enumerate(child_groups):
            if len(group) > 1:  # Only fold if there are multiple nodes
                if debug:
                    group_ids = [node.id for node in group]
                    logger.info(f"[Depth {current_depth}] Folding group {i} with nodes {group_ids}...")
                
                before_nodes = len(result.get_all_nodes())
                result = shallow_fold(result, group)
                after_nodes = len(result.get_all_nodes())
                
                debug_stats["folds_performed"] += 1
                
                if debug:
                    logger.info(f"[Depth {current_depth}] Fold complete. Nodes: {before_nodes} -> {after_nodes}")
    
    if debug:
        elapsed = log_step_time("folding", step_start)
        logger.info(f"[Depth {current_depth}] All folding complete in {elapsed:.2f}s")
        logger.info(f"[Depth {current_depth}] Root now has {len(result.root.children)} children")
    
    # Step 4: Recursively apply MakeFair to each child of root
    step_start = time.time()
    if debug:
        logger.info(f"[Depth {current_depth}] Starting recursion on {len(result.root.children)} children...")
    
    for i, child in enumerate(result.root.children.copy()):  # Use copy to avoid modification issues
        # Skip if child is a leaf or too small
        if child.is_leaf():
            if debug:
                logger.info(f"[Depth {current_depth}] Child {i} (id={child.id}) is a leaf, skipping")
            continue
            
        if child.size < max(h, 1/(2*epsilon)):
            if debug:
                logger.info(f"[Depth {current_depth}] Child {i} (id={child.id}) size {child.size} is too small, skipping")
            continue
        
        if debug:
            logger.info(f"[Depth {current_depth}] Processing child {i} (id={child.id}) with size {child.size}...")
        
        # Create a sub-hierarchy rooted at this child
        sub_hierarchy = Hierarchy(root=child)
        
        # Track node before recursion
        debug_stats["nodes_processed"] += 1
        
        # Recursively apply MakeFair
        new_sub = make_fair(sub_hierarchy, h, k, epsilon, color_ids, max_depth, debug)
        
        # Check if the child was modified
        if new_sub.root != child:
            # Find the child's index in the current list (may have changed)
            try:
                child_idx = result.root.children.index(child)
                result.root.children[child_idx] = new_sub.root
                new_sub.root.parent = result.root
                
                if debug:
                    logger.info(f"[Depth {current_depth}] Child {i} replaced with new subtree")
            except ValueError:
                if debug:
                    logger.warning(f"[Depth {current_depth}] Could not find child {i} in root's children!")
    
    if debug:
        elapsed = log_step_time("recursion", step_start)
        logger.info(f"[Depth {current_depth}] Recursion complete in {elapsed:.2f}s")
    
    # Make sure sizes and color counts are up-to-date
    step_start = time.time()
    result.update_all_sizes()
    _assign_colors(result, color_ids)
    
    if debug:
        elapsed = log_step_time("update", step_start)
        logger.info(f"[Depth {current_depth}] Final updates complete in {elapsed:.2f}s")
        logger.info(f"[Depth {current_depth}] MakeFair complete for this level")
        
        # Print summary statistics
        if current_depth == 1:  # Root level
            total_time = time.time() - debug_stats["start_time"]
            logger.info("-" * 50)
            logger.info(f"MakeFair Overall Statistics:")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Max recursion depth: {debug_stats['max_recursion_depth']}")
            logger.info(f"Nodes processed: {debug_stats['nodes_processed']}")
            logger.info(f"Folds performed: {debug_stats['folds_performed']}")
            logger.info(f"Step times:")
            for step, step_time in debug_stats["step_times"].items():
                logger.info(f"  - {step}: {step_time:.2f}s ({step_time/total_time*100:.1f}%)")
            logger.info("-" * 50)
    
    # Decrement recursion depth before returning
    debug_stats["recursion_depth"] -= 1
    
    return result


def _assign_colors(hierarchy: Hierarchy, color_ids: List[List[int]]) -> None:
    """Assign color counts to all nodes in the hierarchy."""
    # Initialize color counts for leaf nodes
    for node in hierarchy.get_all_nodes():
        node.color_counts = {}
        
        if node.is_leaf():
            # Check which color group this leaf belongs to
            for color_idx, ids in enumerate(color_ids):
                if node.id in ids:
                    node.color_counts[color_idx] = 1
                    break
    
    # Propagate color counts upward
    def update_node_colors(node: Node) -> Dict[int, int]:
        if node.is_leaf():
            return node.color_counts
        
        # Aggregate color counts from children
        aggregate = {}
        for child in node.children:
            child_colors = update_node_colors(child)
            for color, count in child_colors.items():
                aggregate[color] = aggregate.get(color, 0) + count
        
        node.color_counts = aggregate
        return aggregate
    
    if hierarchy.root:
        update_node_colors(hierarchy.root)


def _sort_children_by_color(node: Node, color_idx: int) -> None:
    """Sort the children of a node by the proportion of a specific color."""
    def get_color_proportion(child: Node) -> float:
        """Get the proportion of the specified color in the child node."""
        if child.size == 0:
            return 0.0
            
        color_count = child.color_counts.get(color_idx, 0)
        return color_count / child.size
    
    # Sort children by color proportion (decreasing)
    node.children.sort(key=get_color_proportion, reverse=True)


def _partition_into_groups(children: List[Node], k: int) -> List[List[Node]]:
    """Partition children into k groups for folding."""
    # Determine the number of nodes per group
    n = len(children)
    if n <= k:
        # If there are fewer children than groups, each group has at most one child
        return [[child] for child in children]
    
    nodes_per_group = n // k
    remainder = n % k
    
    # Create groups by partitioning the sorted children
    groups = []
    start = 0
    
    for i in range(k):
        # Add an extra node to early groups if there's a remainder
        group_size = nodes_per_group + (1 if i < remainder else 0)
        end = start + group_size
        
        if start < n:
            groups.append(children[start:end])
        
        start = end
    
    return groups