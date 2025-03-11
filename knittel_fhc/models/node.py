"""
models/node.py
Author: Param Kapur 
Date: 5/10/25
Sources:
 - Knittel et. al. (https://doi.org/10.48550/arXiv.2311.12501)


Defines the Node class, representing a cluster or subtree within a hierarchical clustering.
"""

from __future__ import annotations
from typing import Optional, List, Dict


class Node:
    """
    A Node in a hierarchical clustering tree.

    Attributes:
        id (Optional[int]): Identifier for this node.
        children (List[Node]): Child nodes. If empty, this is a leaf.
        parent (Optional[Node]): Parent node reference (None if root).
        data_indices (List[int]): Indices (in the dataset) of points belonging to this node’s cluster.
        size (int): Number of leaf nodes in the subtree.
        color_counts (Optional[Dict[int, int]]): Tracks how many points of each color 
            are in the subtree (e.g. {0: count_blue, 1: count_red}). None if not used.
    """

    def __init__(
        self,
        node_id: Optional[int] = None,
        children: Optional[List[Node]] = None,
        parent: Optional[Node] = None,
        data_indices: Optional[List[int]] = None,
        size: int = 0,
        color_counts: Optional[Dict[int, int]] = None
    ) -> None:
        self.id: Optional[int] = node_id
        self.children: List[Node] = children if children is not None else []
        self.parent: Optional[Node] = parent
        self.data_indices: List[int] = data_indices if data_indices is not None else []
        self.size: int = size
        self.color_counts: Optional[Dict[int, int]] = color_counts

        # Ensure each child's parent is this node
        for child in self.children:
            child.parent = self

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(self, child_node: Node) -> None:
        self.children.append(child_node)
        child_node.parent = self

    def remove_child(self, child_node: Node) -> None:
        self.children.remove(child_node)
        child_node.parent = None

    def get_leaf_nodes(self) -> List[Node]:
        if self.is_leaf():
            return [self]

        leaves: List[Node] = []
        for child in self.children:
            leaves.extend(child.get_leaf_nodes())
        return leaves

    def update_size(self) -> int:
        """
        Recursively compute and update 'size' (number of leaf nodes) for this node's subtree.

        Returns:
            int: The computed size of this node's subtree.
        """
        if self.is_leaf():
            # If a leaf, size is the number of data_indices if present, else 1
            self.size = len(self.data_indices) if self.data_indices else 1
            return self.size

        total: int = 0
        for child in self.children:
            total += child.update_size()
        self.size = total
        return self.size

    def update_color_counts(self) -> Dict[int, int]:
        """
        Recursively compute and update 'color_counts' for this node's subtree.

        Returns:
            Dict[int, int]: The updated color_counts for this node.
        """
        if self.is_leaf():
            # If leaf, use existing color_counts or create an empty dict
            if self.color_counts is None:
                # if no data, no color; if data, color assignment must be done externally
                self.color_counts = {}
            return self.color_counts

        aggregate: Dict[int, int] = {}
        for child in self.children:
            child_cc = child.update_color_counts()
            for col_val, cnt in child_cc.items():
                aggregate[col_val] = aggregate.get(col_val, 0) + cnt

        self.color_counts = aggregate
        return aggregate

    def copy(self) -> Node:
        """
        Create a deep copy of this node and all descendants.
        The copy's parent will be None.

        Returns:
            Node: A new Node that is a deep copy of this node.
        """
        new_cc: Optional[Dict[int, int]] = None
        if self.color_counts is not None:
            new_cc = dict(self.color_counts)  # shallow copy is fine for int->int map

        new_node = Node(
            node_id=self.id,
            parent=None,
            data_indices=self.data_indices[:],
            size=self.size,
            color_counts=new_cc
        )
        for child in self.children:
            child_copy = child.copy()
            new_node.add_child(child_copy)
        return new_node

    def __repr__(self) -> str:
        """
        String representation for debugging.
        """
        return (f"Node(id={self.id}, "
                f"children={len(self.children)}, "
                f"size={self.size}, "
                f"color_counts={self.color_counts})")