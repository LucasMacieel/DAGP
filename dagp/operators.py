"""Five dimension-preserving neighbourhood operators (§3.3)."""

from __future__ import annotations

from itertools import product as iterproduct
from typing import Optional

from dagp.units import UnitSig, mul_sig, pow_sig, DIMENSIONLESS
from dagp.expression import Node, Op, MAX_TREE_SIZE
from dagp.initialization import EXP_RANGE


_SUBTREE_CACHE: dict[tuple, list[Node]] = {}


def _generate_subtrees_with_sig(
    target_sig: UnitSig,
    var_names: list[str],
    var_units: list[UnitSig],
    exp_range: range = EXP_RANGE,
) -> list[Node]:
    """Generate all monomial subtrees matching a given unit signature."""
    cache_key = (target_sig, tuple(var_names), tuple(var_units), tuple(exp_range))
    if cache_key in _SUBTREE_CACHE:
        return _SUBTREE_CACHE[cache_key]

    results = []
    for exponents in iterproduct(exp_range, repeat=len(var_names)):
        sig = UnitSig(0, 0, 0, 0, 0)
        for exp, unit in zip(exponents, var_units):
            sig = mul_sig(sig, pow_sig(unit, exp))
        if sig == target_sig:
            # Build subtree
            terms = []
            for i, (name, unit, exp) in enumerate(zip(var_names, var_units, exponents)):
                terms.append(Node.variable(name, i, exp, unit))
            if not terms:
                # All-zero = constant 1 (dimensionless)
                if target_sig == DIMENSIONLESS:
                    results.append(Node.constant(1))
                continue
            tree = terms[0]
            for t in terms[1:]:
                tree = Node.operator(Op.MUL, tree, t)
            results.append(tree)

    _SUBTREE_CACHE[cache_key] = results
    return results


def generate_all_neighbours(
    tree: Node,
    var_names: list[str],
    var_units: list[UnitSig],
) -> list[Node]:
    neighbours = []
    subtrees = tree.all_subtrees()

    for st_idx, subtree in enumerate(subtrees):
        st_sig = subtree.unit_sig
        st_size = subtree.size()
        tree_size = tree.size()

        # --- Operator 1: Replacement ---
        # Replace subtree with another one having the same signature
        replacements = _generate_subtrees_with_sig(st_sig, var_names, var_units)
        for repl in replacements:
            if repl.tree_hash() == subtree.tree_hash():
                continue  # skip identity
            new_size = tree_size - st_size + repl.size()
            if new_size > MAX_TREE_SIZE:
                continue
            new_tree = _replace_subtree(tree, subtree, repl)
            if new_tree is not None:
                neighbours.append(new_tree)

        # --- Operator 2: Multiply by integer ---
        # Replace subtree t with (t * k), k ∈ [-3,3]\{0}
        for k in range(-3, 4):
            new_size = tree_size + 2  # operator node + constant node
            if new_size > MAX_TREE_SIZE:
                continue
            const_node = Node.constant(k)
            new_sub = Node.operator(Op.MUL, subtree.copy(), const_node)
            new_tree = _replace_subtree(tree, subtree, new_sub)
            if new_tree is not None:
                neighbours.append(new_tree)

        # --- Operator 3: Divide by integer ---
        for k in range(-3, 4):
            if k == 0 or k == 1:
                continue
            new_size = tree_size + 2
            if new_size > MAX_TREE_SIZE:
                continue
            const_node = Node.constant(k)
            new_sub = Node.operator(Op.DIV, subtree.copy(), const_node)
            new_tree = _replace_subtree(tree, subtree, new_sub)
            if new_tree is not None:
                neighbours.append(new_tree)

        # --- Operator 4: Add commensurate value ---
        commensurate_subs = _generate_subtrees_with_sig(st_sig, var_names, var_units)
        for q in commensurate_subs:
            new_size = tree_size + 1 + q.size()
            if new_size > MAX_TREE_SIZE:
                continue
            new_sub = Node.operator(Op.ADD, subtree.copy(), q)
            new_tree = _replace_subtree(tree, subtree, new_sub)
            if new_tree is not None:
                neighbours.append(new_tree)

        # --- Operator 5: Subtract commensurate value ---
        for q in commensurate_subs:
            new_size = tree_size + 1 + q.size()
            if new_size > MAX_TREE_SIZE:
                continue
            new_sub = Node.operator(Op.SUB, subtree.copy(), q)
            new_tree = _replace_subtree(tree, subtree, new_sub)
            if new_tree is not None:
                neighbours.append(new_tree)

    return neighbours


def _replace_subtree(tree: Node, target: Node, replacement: Node) -> Optional[Node]:
    if tree is target:
        return replacement.copy()
    if tree.is_leaf():
        return tree.copy()
    new_tree = Node()
    new_tree.op = tree.op
    new_tree.left = _replace_subtree(tree.left, target, replacement)
    new_tree.right = _replace_subtree(tree.right, target, replacement)
    if new_tree.left is None or new_tree.right is None:
        return None
    # Recompute unit signature
    if new_tree.op == Op.MUL:
        new_tree.unit_sig = mul_sig(new_tree.left.unit_sig, new_tree.right.unit_sig)
    elif new_tree.op == Op.DIV:
        from dagp.units import div_sig

        new_tree.unit_sig = div_sig(new_tree.left.unit_sig, new_tree.right.unit_sig)
    elif new_tree.op in (Op.ADD, Op.SUB):
        new_tree.unit_sig = new_tree.left.unit_sig
    return new_tree
