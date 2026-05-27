"""Initialisation procedure (§3.2): enumerate valid monomial expressions matching the target unit signature."""

from __future__ import annotations

from itertools import product
from typing import Optional

from dagp.units import UnitSig, mul_sig, pow_sig
from dagp.expression import Node, Op


# Default exponent range
EXP_RANGE = range(-3, 4)  # [-3, -2, -1, 0, 1, 2, 3]


def _signature_for_exponents(
    exponents: tuple[int, ...],
    var_units: list[UnitSig],
) -> UnitSig:
    """Compute the unit signature for x1^e1 * x2^e2 * ... * xp^ep."""
    result = UnitSig(0, 0, 0, 0, 0)
    for exp, unit in zip(exponents, var_units):
        result = mul_sig(result, pow_sig(unit, exp))
    return result


def _build_tree_from_exponents(
    exponents: tuple[int, ...],
    var_names: list[str],
    var_units: list[UnitSig],
) -> Optional[Node]:
    # Group variables into positive and negative exponents
    pos_terms: list[Node] = []
    neg_terms: list[Node] = []

    for i, (name, unit, exp) in enumerate(zip(var_names, var_units, exponents)):
        if exp > 0:
            var_node = Node.variable(name, i, 1, unit)
            for _ in range(exp):
                pos_terms.append(var_node.copy())
        elif exp < 0:
            var_node = Node.variable(name, i, 1, unit)
            for _ in range(-exp):
                neg_terms.append(var_node.copy())

    if not pos_terms and not neg_terms:
        return None

    # Build positive tree: chain with multiplication
    if pos_terms:
        pos_tree = pos_terms[0]
        for t in pos_terms[1:]:
            pos_tree = Node.operator(Op.MUL, pos_tree, t)
    else:
        pos_tree = None

    # Build negative tree: chain with multiplication
    if neg_terms:
        neg_tree = neg_terms[0]
        for t in neg_terms[1:]:
            neg_tree = Node.operator(Op.MUL, neg_tree, t)
    else:
        neg_tree = None

    # Combine pos_tree and neg_tree
    if pos_tree and neg_tree:
        return Node.operator(Op.DIV, pos_tree, neg_tree)
    elif pos_tree:
        return pos_tree
    else:
        # neg_tree only: 1 / neg_tree
        return Node.operator(Op.DIV, Node.constant(1), neg_tree)


def generate_initial_solutions(
    var_names: list[str],
    var_units: list[UnitSig],
    target_unit: UnitSig,
    exp_range: range = EXP_RANGE,
) -> list[Node]:
    n_vars = len(var_names)
    solutions = []

    for exponents in product(exp_range, repeat=n_vars):
        sig = _signature_for_exponents(exponents, var_units)
        if sig == target_unit:
            tree = _build_tree_from_exponents(exponents, var_names, var_units)
            if tree is not None:
                solutions.append(tree)

    return solutions


def count_valid_initializations(
    var_units: list[UnitSig],
    target_unit: UnitSig,
    exp_range: range = EXP_RANGE,
) -> int:
    """Count how many valid initial solutions exist (without building trees)."""
    count = 0
    for exponents in product(exp_range, repeat=len(var_units)):
        sig = _signature_for_exponents(exponents, var_units)
        if sig == target_unit:
            # Exclude all-zero exponents
            if any(e != 0 for e in exponents):
                count += 1
    return count
