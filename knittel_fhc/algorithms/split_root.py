# ── algorithms/split_root.py ────────────────────────────────────────────────
"""
Split‑Root procedure from Knittel et al. (fair hierarchical clustering).

The function `split_root` takes an *arbitrary* binary hierarchy, makes a
deep copy, then transforms the copy so the **root** has (exactly) `h`
children whose sizes all lie in
      [(1/h‑ε)·n , (1/h+ε)·n]
where n is the number of leaves in the entire tree.

Correctness over speed: the code follows the reference algorithm line‑by‑line
with simple list operations—no heavy optimisation.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from models.node import Node
from models.hierarchy import Hierarchy


# ───────────────────────────  small utilities  ────────────────────────────
def _list_leaves(node: Node) -> List[Node]:
    return node.get_leaf_nodes()


def _update_counts(node: Node) -> int:
    """
    Recompute `.size` for every node in the subtree and return the size.
    """
    if node.is_leaf():
        node.size = 1 if node.data_indices else max(node.size, 1)
        return node.size
    total = 0
    for ch in node.children:
        total += _update_counts(ch)
    node.size = total
    return total


def _get_max_id(node: Node) -> int:
    """
    DFS to fetch the current maximal node.id (ignores None ids).
    """
    max_id = -1 if node.id is None else node.id
    for ch in node.children:
        max_id = max(max_id, _get_max_id(ch))
    return max_id


def _order_children(node: Node) -> None:
    """
    Sort children **ascending** by size, then recurse.
    """
    if node.is_leaf():
        return
    node.children.sort(key=lambda c: c.size)  # smallest → first
    for ch in node.children:
        _order_children(ch)


def _patch_compression(node: Node) -> None:
    """
    Remove unary internal nodes produced by tree surgery.
    """
    if node.is_leaf():
        return
    if node.size == 1 and len(node.children) == 1:
        # replace `node` with its single child in the parent’s list
        child = node.children[0]
        parent = node.parent
        if parent is not None:
            parent.children.remove(node)
            parent.children.append(child)
            child.parent = parent
    for ch in list(node.children):  # copy – list may mutate
        _patch_compression(ch)


def _del_ins(root: Node, u: Node, v: Node) -> None:
    """
    Delete subtree v from its parent and **insert** it as sibling of u
    under a freshly created internal node.
    Mirrors the `del_ins` of the reference implementation.
    """
    # 1. remove v from its parent (keep its sibling)
    v_parent = v.parent
    v_sibling: Optional[Node] = None
    for ch in v_parent.children:
        if ch is not v:
            v_sibling = ch
    v_parent.children = [v_sibling]
    if not v_sibling.is_leaf():
        # contract degree‑1 internal node
        v_grand = v_parent.parent
        v_sibling.parent = v_grand
        if v_grand is not None:
            v_grand.children.remove(v_parent)
            v_grand.children.append(v_sibling)

    # 2. create new parent for v and u
    new_id = _get_max_id(root) + 1
    grand = u.parent
    new_node = Node(
        node_id=new_id,
        children=[u, v],
        parent=grand,
        size=u.size + v.size,
    )
    u.parent = new_node
    v.parent = new_node

    # replace u by new_node in grand’s child list
    grand.children.remove(u)
    grand.children.append(new_node)


# ───────────────────────────  main algorithm  ─────────────────────────────
def split_root(
    hierarchy: Hierarchy, h: int, eps: float, debug: bool = False
) -> Hierarchy:
    """
    Return a **new** hierarchy whose root is split into ≤ h balanced pieces.

    Args
    ----
    hierarchy : Hierarchy
        Original binary dendrogram (NOT mutated).
    h : int
        Desired number of root children.
    eps : float
        Allowed size deviation (0 < eps < 1/6 in the paper).
    debug : bool
        Print progress info to stdout.

    Returns
    -------
    Hierarchy
        A deep‑copied hierarchy after the Split‑Root transformation.
    """
    if hierarchy.is_empty():
        return Hierarchy(root=None)

    # Work on a deep copy to preserve the original tree
    H: Hierarchy = hierarchy.copy()
    root: Node = H.root

    # Make sure .size fields are consistent
    _update_counts(root)

    n: int = root.size

    # ── Step 0 : pad with empty children so root has exactly h children
    needed = h - len(root.children)
    while needed > 0:
        new_id = _get_max_id(root) + 1
        null_child = Node(node_id=new_id, parent=root, size=0)
        root.children.append(null_child)
        needed -= 1

    _order_children(root)

    while True:
        vmin = root.children[0]  # smallest
        vmax = root.children[-1]  # largest

        if (vmin.size >= (1 / h - eps) * n) and (vmax.size <= (1 / h + eps) * n):
            # Balanced ⇒ done
            break

        # imbalance amounts
        d1 = 1 / h - vmin.size / n
        d2 = vmax.size / n - 1 / h
        delta = min(d1, d2)

        # ---- find v  (descend along "largest child" chain)
        v = vmax
        while (not v.is_leaf()) and (v.size > delta * n):
            _order_children(v)
            v = v.children[-1]  # keep going into the largest

        # ---- find u  (descend into smallest chain opposite side)
        u = vmin
        while (not u.is_leaf()) and (u.children[-1].size >= v.size):
            _order_children(u)
            u = u.children[0]

        if u.is_leaf() and v.is_leaf():
            # Cannot move leaves any further
            break

        # perform delete‑insert
        if debug:
            print(
                f" Moving subtree v(id={v.id},|v|={v.size}) "
                f"to join u(id={u.id},|u|={u.size})"
            )
        _del_ins(root, u, v)
        _patch_compression(root)
        _order_children(root)
        _update_counts(root)

    return H
