import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict
import random
import time
import os
import math
import copy


class TreeNode:
    """
    TreeNode class for representing hierarchical clustering.
    Each node represents a cluster, with internal nodes representing merged clusters.
    """
    def __init__(self, id=None):
        self.id = id  # Identifier for the node
        self.children = []  # Child nodes
        self.parent = None  # Parent node
        self.leaves = []  # List of leaf indices in this subtree
        self.color_counts = {}  # Counts of each color in the subtree
    
    def add_child(self, node):
        """Add a child to this node"""
        if node is None:
            return
        self.children.append(node)
        node.parent = self
    
    def add_leaf(self, leaf_idx, color=None):
        """Add a leaf to this node"""
        self.leaves.append(leaf_idx)
        if color is not None:
            if color not in self.color_counts:
                self.color_counts[color] = 0
            self.color_counts[color] += 1
    
    def remove_child(self, node):
        """Remove a child from this node"""
        if node in self.children:
            self.children.remove(node)
            node.parent = None
    
    def size(self):
        """Return the size of this subtree (number of leaves)"""
        return len(self.leaves)
    
    def color_fraction(self, color):
        """Return the fraction of leaves with the given color"""
        if self.size() == 0:
            return 0
        return self.color_counts.get(color, 0) / self.size()
    
    def is_leaf(self):
        """Check if this node is a leaf"""
        return len(self.children) == 0 and len(self.leaves) == 1


class HierarchicalClustering:
    """
    Class for representing and manipulating hierarchical clusterings.
    Implements the tree operations and algorithms from the paper.
    """
    def __init__(self, data=None, colors=None):
        """Initialize with data points and their colors"""
        self.data = data
        self.colors = colors if colors is not None else []
        self.root = None
        self.nodes = {}  # Map from node id to TreeNode
        self.leaf_to_node = {}  # Map from leaf index to its TreeNode
        
    def build_from_linkage(self, Z):
        """Build hierarchical clustering from scipy linkage matrix Z"""
        if self.data is None:
            return
            
        n = len(self.data)
        
        # Create leaf nodes
        for i in range(n):
            node = TreeNode(id=i)
            node.add_leaf(i, self.colors[i] if self.colors else None)
            self.nodes[i] = node
            self.leaf_to_node[i] = node
        
        # Build tree from linkage matrix
        for i, (c1, c2, dist, size) in enumerate(Z):
            c1, c2 = int(c1), int(c2)
            new_id = n + i
            new_node = TreeNode(id=new_id)
            
            # Add children
            new_node.add_child(self.nodes[c1])
            new_node.add_child(self.nodes[c2])
            
            # Update leaves and color counts
            new_node.leaves = self.nodes[c1].leaves + self.nodes[c2].leaves
            for color, count in self.nodes[c1].color_counts.items():
                if color not in new_node.color_counts:
                    new_node.color_counts[color] = 0
                new_node.color_counts[color] += count
            for color, count in self.nodes[c2].color_counts.items():
                if color not in new_node.color_counts:
                    new_node.color_counts[color] = 0
                new_node.color_counts[color] += count
            
            self.nodes[new_id] = new_node
        
        # Set root
        self.root = self.nodes[n + len(Z) - 1] if n + len(Z) - 1 in self.nodes else None
        
    def build_from_agglomerative(self):
        """Build using sklearn's AgglomerativeClustering"""
        if self.data is None:
            return
        
        # Perform agglomerative clustering
        try:
            # Convert to linkage matrix format
            Z = linkage(self.data, method='average')
            self.build_from_linkage(Z)
        except Exception as e:
            print(f"Error in build_from_agglomerative: {e}")
            # Create a simple tree if linkage fails
            self._ensure_valid_tree()
    
    def lowest_common_ancestor(self, leaf1, leaf2):
        """Find the lowest common ancestor of two leaves"""
        if leaf1 not in self.leaf_to_node or leaf2 not in self.leaf_to_node:
            return None
        
        node1 = self.leaf_to_node[leaf1]
        node2 = self.leaf_to_node[leaf2]
        
        if node1 is None or node2 is None:
            return None
        
        # Get path from node1 to root
        path1 = set()
        current = node1
        while current:
            path1.add(current)
            current = current.parent
        
        # Find first node in path from node2 to root that is also in path1
        current = node2
        while current and current not in path1:
            current = current.parent
        
        return current
    
    def calculate_cost(self):
        """Calculate Dasgupta's cost function for the hierarchical clustering"""
        if self.data is None or self.root is None:
            return 0
            
        n = len(self.data)
        total_cost = 0
        
        # Calculate pairwise similarities
        similarities = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                # Use Euclidean distance as similarity (invert it)
                dist = np.linalg.norm(self.data[i] - self.data[j])
                similarity = 1 / (1 + dist)  # Transform to similarity
                similarities[i, j] = similarities[j, i] = similarity
        
        # Calculate cost for each pair of points
        for i in range(n):
            for j in range(i+1, n):
                lca = self.lowest_common_ancestor(i, j)
                if lca is None:
                    continue
                cluster_size = len(lca.leaves)
                total_cost += similarities[i, j] * cluster_size
        
        return total_cost
    
    def is_fair(self, alphas, betas):
        """
        Check if the hierarchical clustering is fair according to the paper's definition.
        For each non-singleton cluster C and every color ℓ, we should have:
        alphaℓ·|C| ≤ ℓ(C) ≤ betaℓ·|C|
        """
        if self.root is None:
            return False
            
        colors = set(self.colors)
        
        def check_node(node):
            if node is None or node.is_leaf():
                return True
            
            size = node.size()
            for color in colors:
                color_count = node.color_counts.get(color, 0)
                if color_count < alphas.get(color, 0) * size or color_count > betas.get(color, 0) * size:
                    return False
            
            return all(check_node(child) for child in node.children)
        
        return check_node(self.root)
    
    def is_eps_relatively_balanced(self, eps):
        """
        Check if the hierarchical clustering is ε-relatively balanced.
        For each vertex v with cv children, each child's size should be within
        (1/cv - ε)|C| and (1/cv + ε)|C| of the parent cluster C's size.
        """
        if self.root is None:
            return False
            
        def check_node(node):
            if node is None or node.is_leaf() or len(node.children) == 0:
                return True
            
            cv = len(node.children)
            node_size = node.size()
            target_size = node_size / cv
            
            for child in node.children:
                child_size = child.size()
                if (child_size < (target_size - eps*node_size) or 
                    child_size > (target_size + eps*node_size)):
                    return False
            
            return all(check_node(child) for child in node.children)
        
        return check_node(self.root)
    
    def del_ins(self, u, v):
        """
        Subtree deletion and insertion operator.
        Deletes subtree at u and inserts it at v.
        """
        # Validate nodes before proceeding
        if u is None or v is None:
            print("Warning: Skipping del_ins because u or v is None")
            return
            
        if u == self.root:
            print("Warning: Skipping del_ins because u is the root")
            return
            
        if v == u:
            print("Warning: Skipping del_ins because v is the same as u")
            return
            
        if v.parent is None:
            print("Warning: Skipping del_ins because v has no parent")
            return
        
        if u.parent is None:
            print("Warning: Skipping del_ins because u has no parent")
            return
            
        if v.parent == u.parent and len(u.parent.children) == 2:
            # Special case: u and v are siblings of a binary parent
            print("Warning: Skipping del_ins because u and v are siblings with binary parent")
            return
        
        # Get u's sibling and parent
        u_parent = u.parent
        u_siblings = [c for c in u_parent.children if c != u]
        
        # Remove u from its parent
        u_parent.remove_child(u)
        
        # If u's parent now has only one child, contract it
        if len(u_parent.children) == 1 and u_parent != self.root:
            sibling = u_parent.children[0]
            u_parent_parent = u_parent.parent
            if u_parent_parent is not None:
                u_parent.remove_child(sibling)
                u_parent_parent.remove_child(u_parent)
                u_parent_parent.add_child(sibling)
        
        # Insert u at v
        v_parent = v.parent
        
        # Create new parent for v and u
        try:
            new_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        except Exception as e:
            print(f"Error getting max node id: {e}")
            new_id = len(self.nodes) + 1
            
        new_parent = TreeNode(id=new_id)
        v_parent.remove_child(v)
        v_parent.add_child(new_parent)
        new_parent.add_child(v)
        new_parent.add_child(u)
        
        # Update new_parent's leaves and color counts
        new_parent.leaves = v.leaves + u.leaves
        for color in set(list(v.color_counts.keys()) + list(u.color_counts.keys())):
            new_parent.color_counts[color] = (
                v.color_counts.get(color, 0) + u.color_counts.get(color, 0)
            )
        
        # Update nodes dictionary
        self.nodes[new_parent.id] = new_parent
    
    def shallow_fold(self, trees):
        """
        Shallow tree folding operator.
        Folds multiple trees with same parent into a single tree.
        """
        if not trees or len(trees) <= 1:
            return
        
        # Filter out None trees
        trees = [t for t in trees if t is not None]
        if len(trees) <= 1:
            return
            
        # Ensure all trees have the same parent
        if trees[0] is None or trees[0].parent is None:
            print("Warning: Cannot fold trees with no parent")
            return
            
        parent = trees[0].parent
            
        # Check that all trees have the same parent
        for t in trees:
            if t is None or t.parent != parent:
                print("Warning: Trees must have the same parent for folding")
                return
        
        # Create a new tree to replace the folded trees
        try:
            new_id = max(self.nodes.keys()) + 1 if self.nodes else 0
        except Exception as e:
            print(f"Error getting max node id for shallow_fold: {e}")
            new_id = len(self.nodes) + 1
            
        new_node = TreeNode(id=new_id)
        
        # Add all children of the folded trees as children of the new tree
        all_children = []
        for tree in trees:
            if tree.children:  # Make sure tree has children before extending
                all_children.extend(tree.children)
            if tree.parent:
                tree.parent.remove_child(tree)
        
        for child in all_children:
            if child is not None:
                new_node.add_child(child)
        
        # Add the new tree to the parent
        if parent is not None:
            parent.add_child(new_node)
        
        # Update new_node's leaves and color counts
        new_node.leaves = []
        for child in all_children:
            if child is not None:
                new_node.leaves.extend(child.leaves)
        
        all_colors = set()
        for child in all_children:
            if child is not None:
                all_colors.update(child.color_counts.keys())
        
        for color in all_colors:
            new_node.color_counts[color] = sum(
                child.color_counts.get(color, 0) for child in all_children if child is not None
            )
        
        # Update nodes dictionary
        self.nodes[new_node.id] = new_node
        
        # If new_node has more than 2 children, binarize it
        if len(new_node.children) > 2:
            self.binarize_node(new_node)
    
    def binarize_node(self, node):
        """Convert a node with more than 2 children to a binary sub-tree"""
        if node is None or len(node.children) <= 2:
            return
        
        children = node.children.copy()
        node.children = []
        
        # Create a right-leaning binary tree
        current = node
        for i, child in enumerate(children):
            if child is None:
                continue
                
            if i == 0:
                current.add_child(child)
            elif i == len(children) - 1:
                current.add_child(child)
            else:
                try:
                    new_id = max(self.nodes.keys()) + 1 if self.nodes else 0
                except Exception as e:
                    print(f"Error getting max node id for binarize: {e}")
                    new_id = len(self.nodes) + 1
                    
                new_node = TreeNode(id=new_id)
                current.add_child(new_node)
                new_node.add_child(child)
                current = new_node
                
                # Update new_node's leaves and color counts
                new_node.leaves = child.leaves.copy()
                new_node.color_counts = copy.deepcopy(child.color_counts)
                
                # Update nodes dictionary
                self.nodes[new_node.id] = new_node
    
    def split_root(self, h, eps):
        """
        Implements Algorithm 1: SplitRoot.
        Balances the root to have h children with approximately equal sizes.
        """
        # Step 1: Initialize
        if self.root is None:
            print("Warning: Hierarchy is empty in split_root")
            self._ensure_valid_tree()
            return
        
        # Step 2-3: Add null children to root until it has h children
        while len(self.root.children) < h:
            try:
                new_id = max(self.nodes.keys()) + 1 if self.nodes else 0
            except Exception as e:
                print(f"Error getting max node id for new child: {e}")
                new_id = len(self.nodes) + 1
                
            new_node = TreeNode(id=new_id)
            self.root.add_child(new_node)
            self.nodes[new_id] = new_node
        
        # Main loop: balance the children
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Step 4-5: Find min and max children
            children = self.root.children
            if not children:
                print("Warning: Root has no children")
                break
                
            v_min = min(children, key=lambda c: c.size())
            v_max = max(children, key=lambda c: c.size())
            
            # Calculate sizes and target size
            n = self.root.size()
            if n == 0:
                print("Warning: Root has size 0")
                break
                
            n_min = v_min.size()
            n_max = v_max.size()
            target = n / h
            
            # Step 6: Check if balanced
            if n_max <= target + eps*n and n_min >= target - eps*n:
                break
                
            # Step 7-9: Calculate deltas and delta
            delta1 = target/n - n_min/n
            delta2 = n_max/n - target/n
            delta = min(delta1, delta2)
            
            # Step 10-14: Find subtree to move from v_max
            v = v_max
            find_v_attempts = 0
            max_v_attempts = 20
            
            while v.size() > delta * n and len(v.children) > 0 and find_v_attempts < max_v_attempts:
                find_v_attempts += 1
                # Find right child (larger child)
                if len(v.children) >= 2:
                    right_child = max(v.children, key=lambda c: c.size())
                    v = right_child
                else:
                    v = v.children[0]
            
            # Step 16-19: Find insertion spot under v_min
            u = v_min
            find_u_attempts = 0
            max_u_attempts = 20
            
            while len(u.children) > 0 and find_u_attempts < max_u_attempts:
                find_u_attempts += 1
                # Find left child (smaller child) that's still larger than v
                left_child = None
                for child in u.children:
                    if left_child is None or child.size() < left_child.size():
                        left_child = child
                
                # Check if right child (complementary to left) is smaller than v
                if len(u.children) >= 2:
                    right_child = max(u.children, key=lambda c: c.size())
                    if right_child.size() < v.size():
                        break
                
                # If we can't find a suitable child, break
                if left_child is None:
                    break
                    
                u = left_child
            
            # If u or v is invalid, try a different approach
            if u is None or v is None or u == v or v == self.root or find_v_attempts >= max_v_attempts or find_u_attempts >= max_u_attempts:
                print(f"Warning: Invalid nodes at iteration {iteration}. Trying a different balancing approach.")
                # Try a simpler approach: just redistribute leaves
                self._simple_balance(h, eps)
                return
            
            # Step 20: Apply deletion and insertion
            # If the operation fails, try another approach
            try:
                self.del_ins(v, u)
            except Exception as e:
                print(f"Error in del_ins: {e}. Trying a different balancing approach.")
                self._simple_balance(h, eps)
                return
                
        if iteration >= max_iterations:
            print("Warning: SplitRoot reached maximum iterations without converging")
            # Fall back to simple balancing
            self._simple_balance(h, eps)
            
    def _simple_balance(self, h, eps):
        """
        A simpler approach to balance the root node when the main algorithm fails.
        Redistributes leaves directly among h children.
        """
        print("Using simple balancing approach")
        
        if self.root is None:
            print("Warning: Root is None in _simple_balance")
            self._ensure_valid_tree()
            return
            
        # Collect all leaves
        all_leaves = []
        for child in self.root.children:
            for leaf_idx in child.leaves:
                if leaf_idx in self.leaf_to_node:
                    all_leaves.append(leaf_idx)
        
        # If no leaves, create some dummy ones
        if not all_leaves and self.data is not None:
            for i in range(min(len(self.data), 10)):
                all_leaves.append(i)
        
        # Clear current children
        self.root.children = []
        
        # Create h new children
        children = []
        for i in range(h):
            try:
                new_id = max(self.nodes.keys()) + 1 if self.nodes else 0
            except Exception as e:
                print(f"Error getting max node id for new balanced child: {e}")
                new_id = len(self.nodes) + 1
                
            new_node = TreeNode(id=new_id)
            children.append(new_node)
            self.nodes[new_id] = new_node
        
        # Distribute leaves evenly
        if all_leaves:
            leaves_per_child = len(all_leaves) // h
            remainder = len(all_leaves) % h
            
            start_idx = 0
            for i, child in enumerate(children):
                # Give extra leaf to first 'remainder' children
                num_leaves = leaves_per_child + (1 if i < remainder else 0)
                end_idx = min(start_idx + num_leaves, len(all_leaves))
                
                child_leaves = all_leaves[start_idx:end_idx]
                for leaf_idx in child_leaves:
                    child.add_leaf(leaf_idx, self.colors[leaf_idx] if self.colors else None)
                    # Create a new leaf node or use existing one
                    if leaf_idx in self.leaf_to_node:
                        leaf_node = self.leaf_to_node[leaf_idx]
                    else:
                        leaf_node = TreeNode(id=leaf_idx)
                        leaf_node.add_leaf(leaf_idx, self.colors[leaf_idx] if self.colors else None)
                        self.nodes[leaf_idx] = leaf_node
                        self.leaf_to_node[leaf_idx] = leaf_node
                
                start_idx = end_idx
                self.root.add_child(child)
        else:
            # If no leaves, just add the children directly
            for child in children:
                self.root.add_child(child)
    
    def make_fair(self, h, k, eps):
        """
        Implements Algorithm 2: MakeFair.
        Creates a fair hierarchical clustering.
        """
        try:
            # Step 1: Apply SplitRoot
            self.split_root(h, eps)
            
            # Get current number of children at root
            h_prime = len(self.root.children) if self.root else 0
            
            # Steps 2-7: Apply shallow tree folding for each color
            unique_colors = set(self.colors)
            for color in unique_colors:
                if not self.root or not self.root.children:
                    continue
                    
                # Step 4: Order children by decreasing color fraction
                ordered_children = sorted(
                    self.root.children,
                    key=lambda c: c.color_fraction(color),
                    reverse=True
                )
                
                # Step 5: Group and fold
                if len(ordered_children) >= k:
                    # Divide into k parts
                    chunk_size = max(1, len(ordered_children) // k)
                    for i in range(min(k, len(ordered_children))):
                        if i * chunk_size >= len(ordered_children):
                            break
                        
                        # Get trees to fold (one from each chunk)
                        trees_to_fold = []
                        for j in range(k):
                            idx = i + j * chunk_size
                            if idx < len(ordered_children):
                                trees_to_fold.append(ordered_children[idx])
                        
                        # Apply shallow fold if we have more than one tree
                        if len(trees_to_fold) > 1:
                            try:
                                self.shallow_fold(trees_to_fold)
                            except Exception as e:
                                print(f"Warning: Error in shallow_fold: {e}")
                
                # Update h_prime after folding
                h_prime = len(self.root.children) if self.root else 0
            
            # Steps 8-14: Recursively apply to children
            if self.root:
                for child in list(self.root.children):  # Copy to avoid modification issues
                    if child and child.size() >= max(1/(2*eps), h):
                        # Create a subproblem
                        sub_hc = HierarchicalClustering()
                        sub_hc.root = child
                        
                        # Create a deep copy of the nodes in this subtree
                        sub_nodes = {}
                        sub_leaves = []
                        
                        def collect_subtree(node):
                            if node is None:
                                return
                            sub_nodes[node.id] = node
                            if node.is_leaf() and node.leaves:
                                sub_leaves.extend(node.leaves)
                            for c in node.children:
                                collect_subtree(c)
                                
                        collect_subtree(child)
                        sub_hc.nodes = sub_nodes
                        
                        # Extract colors for the subproblem
                        sub_colors = []
                        for leaf_idx in sub_leaves:
                            if leaf_idx < len(self.colors):
                                sub_colors.append(self.colors[leaf_idx])
                            else:
                                # Fallback for missing color
                                sub_colors.append(0)
                                
                        sub_hc.colors = sub_colors
                        
                        # Update leaf_to_node mapping
                        for leaf_idx in sub_leaves:
                            if leaf_idx in self.leaf_to_node:
                                sub_hc.leaf_to_node[leaf_idx] = self.leaf_to_node[leaf_idx]
                        
                        # Recursively apply MakeFair
                        try:
                            sub_hc.make_fair(h, k, eps)
                            
                            # Update the original hierarchy
                            self._update_from_subproblem(child, sub_hc)
                        except Exception as e:
                            print(f"Warning: Error in recursive make_fair: {e}")
                    else:
                        # Replace with a flat tree (depth 1)
                        try:
                            self._flatten_subtree(child)
                        except Exception as e:
                            print(f"Warning: Error in _flatten_subtree: {e}")
        
        except Exception as e:
            print(f"Warning: Error in make_fair: {e}")
            # At this point, try to ensure we have a valid tree even if the algorithm fails
            self._ensure_valid_tree()
    
    def _get_ancestors(self, node):
        """Get all ancestors of a node"""
        ancestors = []
        current = node.parent if node else None
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors
    
    def _update_from_subproblem(self, original_node, sub_hc):
        """Update the original hierarchy with the modified subproblem"""
        if original_node is None or sub_hc.root is None:
            return
            
        # Update nodes dictionary
        for node_id, node in sub_hc.nodes.items():
            if node_id not in self.nodes:
                self.nodes[node_id] = node
        
        # Replace original_node with sub_hc.root
        if original_node.parent:
            try:
                idx = original_node.parent.children.index(original_node)
                original_node.parent.children[idx] = sub_hc.root
                sub_hc.root.parent = original_node.parent
            except (ValueError, IndexError) as e:
                print(f"Warning: Error replacing node in _update_from_subproblem: {e}")
                original_node.parent.add_child(sub_hc.root)
                original_node.parent.remove_child(original_node)
    
    def _flatten_subtree(self, node):
        """Replace a subtree with a flat tree (depth 1)"""
        if node is None:
            return
            
        # Create a new node with the same properties
        flat_node = TreeNode(id=node.id)
        flat_node.leaves = node.leaves.copy()
        flat_node.color_counts = copy.deepcopy(node.color_counts)
        flat_node.parent = node.parent
        
        # Replace node in its parent's children
        if node.parent:
            try:
                idx = node.parent.children.index(node)
                node.parent.children[idx] = flat_node
            except (ValueError, IndexError) as e:
                print(f"Warning: Error replacing node in _flatten_subtree: {e}")
                node.parent.add_child(flat_node)
                node.parent.remove_child(node)
        
        # Create leaf nodes for each leaf
        for leaf_idx in node.leaves:
            try:
                new_id = max(self.nodes.keys()) + 1 if self.nodes else 0
            except Exception as e:
                print(f"Error getting max node id for leaf: {e}")
                new_id = len(self.nodes) + 1
                
            leaf_node = TreeNode(id=new_id)
            leaf_node.add_leaf(leaf_idx, self.colors[leaf_idx] if self.colors and leaf_idx < len(self.colors) else None)
            flat_node.add_child(leaf_node)
            self.nodes[new_id] = leaf_node
            if leaf_idx < len(self.data) if self.data is not None else True:
                self.leaf_to_node[leaf_idx] = leaf_node
        
        # Update nodes dictionary
        self.nodes[flat_node.id] = flat_node
        
        # If this was the root, update the root reference
        if node == self.root:
            self.root = flat_node
            
    def _ensure_valid_tree(self):
        """
        Ensures we have a valid tree structure even if algorithms fail.
        Creates a simple balanced tree if necessary.
        """
        # Check if we have a valid root
        if self.root is None or len(self.root.leaves) == 0:
            if not self.data or len(self.data) == 0:
                print("No data available to create a tree")
                return
                
            # Create a simple balanced tree
            print("Creating fallback balanced tree")
            n = len(self.data)
            
            # Create leaf nodes
            leaf_nodes = []
            for i in range(n):
                node = TreeNode(id=i)
                color = self.colors[i] if self.colors and i < len(self.colors) else None
                node.add_leaf(i, color)
                self.nodes[i] = node
                self.leaf_to_node[i] = node
                leaf_nodes.append(node)
                
            # Create a balanced binary tree
            while len(leaf_nodes) > 1:
                new_nodes = []
                for i in range(0, len(leaf_nodes), 2):
                    if i + 1 < len(leaf_nodes):
                        # Create a parent for two nodes
                        try:
                            parent_id = max(self.nodes.keys()) + 1 if self.nodes else 0
                        except Exception as e:
                            print(f"Error getting max node id for parent: {e}")
                            parent_id = len(self.nodes) + 1
                            
                        parent = TreeNode(id=parent_id)
                        parent.add_child(leaf_nodes[i])
                        parent.add_child(leaf_nodes[i+1])
                        parent.leaves = leaf_nodes[i].leaves + leaf_nodes[i+1].leaves
                        
                        # Update color counts
                        for color in set(list(leaf_nodes[i].color_counts.keys()) + list(leaf_nodes[i+1].color_counts.keys())):
                            if color not in parent.color_counts:
                                parent.color_counts[color] = 0
                            parent.color_counts[color] += leaf_nodes[i].color_counts.get(color, 0) + leaf_nodes[i+1].color_counts.get(color, 0)
                        
                        self.nodes[parent_id] = parent
                        new_nodes.append(parent)
                    else:
                        # Add odd node directly to next level
                        new_nodes.append(leaf_nodes[i])
                
                leaf_nodes = new_nodes
                
            if leaf_nodes:
                self.root = leaf_nodes[0]
            return
            
        # Check if any node has null references
        for node_id, node in list(self.nodes.items()):
            if node.parent is not None and node.parent.id not in self.nodes:
                node.parent = None
                
        # Ensure all nodes are reachable from root
        visited = set()
        
        def visit(node):
            if node is None:
                return
            visited.add(node.id)
            for child in node.children:
                visit(child)
                
        visit(self.root)
        
        # Remove unreachable nodes
        for node_id in list(self.nodes.keys()):
            if node_id not in visited:
                del self.nodes[node_id]


def generate_colored_data(n_samples=300, n_features=2, n_colors=2, centers=4, random_state=42):
    """
    Generate synthetic data with specified number of colors.
    The colors will be assigned to ensure some degree of clustering.
    """
    # Generate clustered data
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, 
                      centers=centers, random_state=random_state)
    
    # Assign colors to make them somewhat grouped
    colors = []
    color_centers = np.random.choice(centers, n_colors, replace=False)
    
    for cluster_id in y:
        if cluster_id in color_centers:
            color_idx = np.where(color_centers == cluster_id)[0][0]
        else:
            # Assign a random color
            color_idx = np.random.randint(0, n_colors)
        colors.append(color_idx)
    
    return X, colors


def visualize_clustering(X, colors, hc, title="Hierarchical Clustering", is_fair=False, show_plot=True, 
                      save_path=None, filename=None):
    """Visualize the data and the hierarchical clustering"""
    plt.figure(figsize=(14, 6))
    
    # Plot the data points colored by their assigned colors
    plt.subplot(1, 2, 1)
    unique_colors = set(colors)
    for color in unique_colors:
        mask = np.array(colors) == color
        plt.scatter(X[mask, 0], X[mask, 1], label=f'Color {color}', alpha=0.7)
    plt.title("Data Points")
    plt.legend()
    
    # Plot the dendrogram
    plt.subplot(1, 2, 2)
    
    # Convert to format for dendrogram visualization
    # We'll need a custom structure for this
    # This is a simplified approach that will show the structure
    def plot_tree(node, x, y, dx, level=0):
        if node is None or node.is_leaf():
            return
        
        # Plot children
        n_children = len(node.children)
        for i, child in enumerate(node.children):
            if child is None:
                continue
            child_x = x + (i - n_children/2 + 0.5) * dx
            child_y = y - 1
            
            # Plot line to child
            plt.plot([x, child_x], [y, child_y], 'k-')
            
            # Plot subtree
            plot_tree(child, child_x, child_y, dx/n_children, level+1)
    
    # Start at root
    if hc.root:
        plot_tree(hc.root, 0, 0, 2)
    plt.axis('off')
    plt.title("Hierarchical Clustering Structure")
    
    plt.suptitle(f"{title} - {'Fair' if is_fair else 'Unfair'}")
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path and filename:
        try:
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path)
        except Exception as e:
            print(f"Error saving figure: {e}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    
    # Always close the figure to prevent memory leaks
    plt.close()


def run_experiment(n_samples=200, n_colors=2, h=4, k=2, eps=0.1, show_plots=True, save_plots=False, 
                save_path=None):
    """Run an experiment with the specified parameters"""
    print(f"Running experiment with n={n_samples}, colors={n_colors}, h={h}, k={k}, eps={eps}")
    
    # Generate data
    X, colors = generate_colored_data(n_samples=n_samples, n_colors=n_colors)
    
    # Evaluate clustering
    vanilla_hc = HierarchicalClustering(X, colors)
    vanilla_hc.build_from_agglomerative()
    
    # Calculate cost of vanilla clustering
    vanilla_cost = vanilla_hc.calculate_cost()
    
    # Check if vanilla clustering is fair
    alphas = {color: 0.3 for color in set(colors)}  # Example fairness parameters
    betas = {color: 0.7 for color in set(colors)}
    vanilla_is_fair = vanilla_hc.is_fair(alphas, betas)
    
    # Visualize vanilla clustering
    if show_plots or save_plots:
        filename = f"vanilla_n{n_samples}_h{h}_k{k}_eps{eps}.png" if save_plots else None
        visualize_clustering(X, colors, vanilla_hc, "Vanilla Hierarchical Clustering", 
                             vanilla_is_fair, show_plot=show_plots, save_path=save_path, 
                             filename=filename)
    
    # Create a copy for fair clustering
    fair_hc = HierarchicalClustering(X, colors)
    fair_hc.build_from_agglomerative()
    
    # Apply MakeFair algorithm
    start_time = time.time()
    try:
        fair_hc.make_fair(h, k, eps)
    except Exception as e:
        print(f"Error in make_fair: {e}")
    end_time = time.time()
    
    # Calculate cost of fair clustering
    fair_cost = fair_hc.calculate_cost()
    
    # Check if fair clustering is actually fair
    fair_is_fair = fair_hc.is_fair(alphas, betas)
    
    # Visualize fair clustering
    if show_plots or save_plots:
        filename = f"fair_n{n_samples}_h{h}_k{k}_eps{eps}.png" if save_plots else None
        visualize_clustering(X, colors, fair_hc, "Fair Hierarchical Clustering", 
                             fair_is_fair, show_plot=show_plots, save_path=save_path, 
                             filename=filename)
    
    # Print comparison results
    print("Evaluation Results:")
    print(f"Vanilla Clustering - Cost: {vanilla_cost:.2f}, Fair: {vanilla_is_fair}")
    print(f"Fair Clustering - Cost: {fair_cost:.2f}, Fair: {fair_is_fair}")
    print(f"Cost ratio (Fair/Vanilla): {fair_cost/vanilla_cost:.2f}")
    print(f"Time to make fair: {end_time - start_time:.2f} seconds")
    
    return vanilla_hc, fair_hc, vanilla_cost, fair_cost


# Example usage
if __name__ == "__main__":
    # Create output directory for saving plots
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run basic experiment with visualization
    vanilla_hc, fair_hc, vanilla_cost, fair_cost = run_experiment(  # TODO: rename this 
        n_samples=4000, 
        n_colors=2, 
        h=8, 
        k=2, 
        eps=0.1,
        show_plots=True,
        save_plots=True,
        save_path=output_dir
    )
    
    # Additional experiment with different parameters (no plots for these)
    print("\nVarying parameters:")
    
    # Try different values of h
    h_results = []
    for h in [8, 16, 32]:
        try:
            _, _, v_cost, f_cost = run_experiment(
                n_samples=2000, h=h, k=2, eps=0.1, 
                show_plots=False
            )
            ratio = f_cost/v_cost
            h_results.append((h, ratio))
            print(f"h={h}: Cost ratio = {ratio:.2f}")
        except Exception as e:
            print(f"Error in h={h} experiment: {e}")
    
    # Try different values of k
    k_results = []
    for k in [2, 3, 4]:
        try:
            _, _, v_cost, f_cost = run_experiment(
                n_samples=2000, h=4, k=k, eps=0.1, 
                show_plots=False
            )
            ratio = f_cost/v_cost
            k_results.append((k, ratio))
            print(f"k={k}: Cost ratio = {ratio:.2f}")
        except Exception as e:
            print(f"Error in k={k} experiment: {e}")
    
    # Try different values of eps
    eps_results = []
    for eps in [0.05, 0.1, 0.2]:
        try:
            _, _, v_cost, f_cost = run_experiment(
                n_samples=2000, h=4, k=2, eps=eps, 
                show_plots=False
            )
            ratio = f_cost/v_cost
            eps_results.append((eps, ratio))
            print(f"eps={eps}: Cost ratio = {ratio:.2f}")
        except Exception as e:
            print(f"Error in eps={eps} experiment: {e}")
    
    # Plot parameter study results if we have results
    if h_results and k_results and eps_results:
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot h parameter results
            plt.subplot(1, 3, 1)
            h_values, h_ratios = zip(*h_results)
            plt.plot(h_values, h_ratios, 'o-', linewidth=2, markersize=8)
            plt.xlabel('h (number of children)')
            plt.ylabel('Cost Ratio (Fair/Vanilla)')
            plt.title('Effect of h on Cost Ratio')
            plt.grid(True)
            
            # Plot k parameter results
            plt.subplot(1, 3, 2)
            k_values, k_ratios = zip(*k_results)
            plt.plot(k_values, k_ratios, 'o-', linewidth=2, markersize=8)
            plt.xlabel('k (fold parameter)')
            plt.ylabel('Cost Ratio (Fair/Vanilla)')
            plt.title('Effect of k on Cost Ratio')
            plt.grid(True)
            
            # Plot eps parameter results
            plt.subplot(1, 3, 3)
            eps_values, eps_ratios = zip(*eps_results)
            plt.plot(eps_values, eps_ratios, 'o-', linewidth=2, markersize=8)
            plt.xlabel('ε (balance parameter)')
            plt.ylabel('Cost Ratio (Fair/Vanilla)')
            plt.title('Effect of ε on Cost Ratio')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "parameter_study.png"))
            plt.show()
            plt.close()
        except Exception as e:
            print(f"Error creating parameter study plot: {e}")
    
    # Run a scaling experiment
    print("\nScaling experiment:")
    sizes = [100, 200, 400, 800]
    scaling_results = []
    
    for size in sizes:
        try:
            _, _, v_cost, f_cost = run_experiment(
                n_samples=size, h=4, k=2, eps=0.1, 
                show_plots=False
            )
            ratio = f_cost/v_cost
            scaling_results.append((size, ratio))
            print(f"n={size}: Cost ratio = {ratio:.2f}")
        except Exception as e:
            print(f"Error in n={size} experiment: {e}")
    
    # Plot scaling results if we have results
    if scaling_results:
        try:
            plt.figure(figsize=(10, 6))
            n_values, n_ratios = zip(*scaling_results)
            plt.plot(n_values, n_ratios, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Dataset Size (n)')
            plt.ylabel('Cost Ratio (Fair/Vanilla)')
            plt.title('Effect of Dataset Size on Cost Ratio')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "scaling_study.png"))
            plt.show()
            plt.close()
        except Exception as e:
            print(f"Error creating scaling plot: {e}")