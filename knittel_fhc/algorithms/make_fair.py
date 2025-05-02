# ── algorithms/make_fair.py ────────────────────────────────────────────────
"""
Straightforward port of the reference MakeFair routine (Knittel et al.)
to the modern Node / Hierarchy classes.

* Two‑colour case only.
* Works **in‑place** on the tree returned by split_root.
"""

from __future__ import annotations
from typing import List

import numpy as np

from models.node import Node
from models.hierarchy import Hierarchy
from algorithms.split_root import split_root


# ───────────────────────── helper utilities ──────────────────────────────
def update_counts(node: Node) -> None:
    """
    Recursively set node.size = #leaves underneath.
    """
    if node.is_leaf():
        node.size = 1
    else:
        node.size = 0
        for ch in node.children:
            update_counts(ch)
            node.size += ch.size


def list_leaves(node: Node) -> List[Node]:
    if node.is_leaf():
        return [node]
    leaves: List[Node] = []
    for ch in node.children:
        leaves.extend(list_leaves(ch))
    return leaves


def update_colors(node: Node, red_ids: set[int], blue_ids: set[int]) -> None:
    """
    Attach/refresh `node.color` = (#red / size) for every node.
    """
    if node.is_leaf():
        if node.id in red_ids:
            node.color = 1.0
        else:
            node.color = 0.0
        return

    red_total = 0.0
    for ch in node.children:
        update_colors(ch, red_ids, blue_ids)
        red_total += ch.color * ch.size
    node.color = red_total / node.size


def order_children(node: Node, sort_by: int = 0) -> None:
    """
    sort_by = 0 → by size  (ascending)
    sort_by = 1 → by red‑fraction (descending)
    sort_by = 2 → by blue‑fraction (descending)
    """
    if node.is_leaf():
        return

    if sort_by == 0:
        node.children.sort(key=lambda x: x.size)
    elif sort_by == 1:
        node.children.sort(key=lambda x: x.color / max(x.size, 1), reverse=True)
    elif sort_by == 2:
        node.children.sort(key=lambda x: (1 - x.color / max(x.size, 1)), reverse=True)

    for ch in node.children:
        order_children(ch, sort_by)


def get_max_id(node: Node) -> int:
    mid = -1 if node.id is None else node.id
    for ch in node.children:
        mid = max(mid, get_max_id(ch))
    return mid


def fold(root: Node, trees: List[Node]) -> None:
    """
    Same as reference: replace every listed subtree (siblings!) by a new node
    whose children are their leaves.
    """
    if not trees:
        return

    parent = trees[0].parent
    new_id = get_max_id(root) + 1
    new_node = Node(node_id=new_id, parent=parent, children=[], size=0)
    parent.children.append(new_node)

    count = 0
    new_children: List[Node] = []
    for t in trees:
        # detach t from parent
        if t in parent.children:
            parent.children.remove(t)
        if t.is_leaf():
            new_children.append(t)
            t.parent = new_node
            count += 1
        else:
            for ch in t.children:
                new_children.append(ch)
                ch.parent = new_node
            count += t.size
        t.parent = None  # detach fully

    new_node.children = new_children
    new_node.size = count
    # colour counts fixed later via update_colors


# ─────────────────────────── MakeFair itself ─────────────────────────────
def make_fair(
    hierarchy: Hierarchy,
    h: int,
    k: int,
    eps: float,
    colour_ids: List[int],
) -> Hierarchy:
    """
    Thin wrapper: balance root with split_root, then run in‑place MakeFair
    exactly as in the original pseudocode.  Returns the **same** hierarchy
    object (for chaining).
    """
    if hierarchy.is_empty():
        return hierarchy

    # Balanced copy (split_root already returns a *new* Hierarchy)
    H = split_root(hierarchy, h, eps)
    root: Node = H.root

    # Build helper id sets
    blue_ids = colour_ids[1]
    red_ids = colour_ids[0]

    # Initial colour + count annotations
    update_colors(root, red_ids, blue_ids)
    update_counts(root)

    # ---------------- Phase 1: root‑level folding ----------------
    for c in range(2):  # red pass, blue pass
        order_children(root, c + 1)  # 1 → red, 2 → blue
        to_fold: List[Node] = []
        for i in range(k):
            for j in range(round(h / k) - 1):
                idx = i + j * k
                if idx < len(root.children):
                    to_fold.append(root.children[idx])
        fold(root, to_fold)
        update_counts(root)
        update_colors(root, red_ids, blue_ids)

    # ---------------- Phase 2: recurse ---------------------------
    thresh = np.max([1 / (2 * eps), h])
    for child in list(root.children):  # list() → safe iter
        if child.size >= thresh:
            # recurse on subtree – wrap in tiny Hierarchy to reuse code
            sub_h = Hierarchy(root=child)
            make_fair(sub_h, h, k, eps, colour_ids)
        else:
            # flatten children to leaves if still internal
            if not child.is_leaf():
                leaves = list_leaves(child)
                child.children = leaves
                for lf in leaves:
                    lf.parent = child
                child.size = len(leaves)

    return H
