"""
models/hierarchy.py
Author: Param Kapur 
Date: 5/10/25
Sources:
 - Knittel et. al. (https://doi.org/10.48550/arXiv.2311.12501)

Defines the Hierarchy class, which wraps a Node as the root of a hierarchical clustering tree.
"""

from __future__ import annotations
from typing import List, Optional
from models.node import Node


class Hierarchy:
    """
    A Hierarchy encapsulates the entire tree for a hierarchical clustering.

    Attributes:
        root (Optional[Node]): The root node of this hierarchical clustering.
    """

    def __init__(self, root: Optional[Node] = None) -> None:
        self.root: Optional[Node] = root

    def is_empty(self) -> bool:
        return self.root is None

    def get_leaves(self) -> List[Node]:
        if self.is_empty():
            return []
        return self.root.get_leaf_nodes()

    def update_all_sizes(self) -> None:
        """
        Recursively update the size (number of leaf nodes) for every node in the hierarchy.
        """
        if not self.is_empty():
            self.root.update_size()

    def update_all_color_counts(self) -> None:
        """
        Recursively update the color_counts for every node in the hierarchy.
        """
        if not self.is_empty():
            self.root.update_color_counts()

    def copy(self) -> Hierarchy:
        """
        Create a deep copy of this entire hierarchy.

        Returns:
            Hierarchy: A new Hierarchy with a copied root node.
        """
        if self.is_empty():
            return Hierarchy(root=None)
        new_root = self.root.copy()
        return Hierarchy(root=new_root)

    def get_all_nodes(self) -> List[Node]:
        """
        Retrieve all nodes (internal + leaves) in this hierarchy using a DFS traversal.

        Returns:
            List[Node]: List of all nodes in the tree.
        """
        if self.is_empty():
            return []

        nodes: List[Node] = []
        stack: List[Node] = [self.root]
        while stack:
            current: Node = stack.pop()
            nodes.append(current)
            stack.extend(current.children)
        return nodes

    def find_node_by_id(self, node_id: int) -> Optional[Node]:
        for node in self.get_all_nodes():
            if node.id == node_id:
                return node
        return None

    def compute_height(self) -> int:
        """
        Compute the height of the hierarchy (longest path from root to leaf).
        TODO: This can probably be done more efficiently

        Returns:
            int: The height of the tree (0 if root is leaf or empty).
        """
        if self.is_empty():
            return 0

        def node_height(n: Node) -> int:
            if n.is_leaf():
                return 0
            return 1 + max(node_height(c) for c in n.children)

        return node_height(self.root)

    def lowest_common_ancestor(self, node_a: Node, node_b: Node) -> Optional[Node]:
        """
        Compute the lowest common ancestor (LCA) of two nodes in this hierarchy.
        TODO: Implement this with Single Travesal i.e. reccur on the left and 
            right subtrees, checking if the root = LCA(a, b)

        Args:
            node_a (Node): First node
            node_b (Node): Second node

        Returns:
            Optional[Node]: The LCA if both nodes in the same tree, else None.
        """
        ancestors_a = []
        current: Optional[Node] = node_a
        while current is not None:
            ancestors_a.append(current)
            current = current.parent

        ancestors_a_set = set(ancestors_a)

        current = node_b
        while current is not None:
            if current in ancestors_a_set:
                return current
            current = current.parent

        return None

    def __repr__(self) -> str:
        if self.is_empty():
            return "Hierarchy(empty=True)"
        return f"Hierarchy(root={self.root})"
