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
    # Collect non-zero terms
    terms: list[Node] = []
    for i, (name, unit, exp) in enumerate(zip(var_names, var_units, exponents)):
        if exp != 0:
            terms.append(Node.variable(name, i, exp, unit))

    if not terms:
        return None  # All zero exponents = constant 1 = not useful

    # Chain terms with multiplication
    tree = terms[0]
    for t in terms[1:]:
        tree = Node.operator(Op.MUL, tree, t)
    return tree


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
