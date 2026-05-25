"""Expression tree representation for DAGP."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from dagp.units import UnitSig, DIMENSIONLESS, mul_sig, div_sig, pow_sig, commensurate


MAX_TREE_SIZE = 42


class Op(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


class Node:
    """Expression tree node."""

    __slots__ = (
        "op",
        "left",
        "right",
        "var_name",
        "var_idx",
        "exponent",
        "const_val",
        "unit_sig",
        "_hash_cache",
    )

    def __init__(self) -> None:
        self.op: Optional[Op] = None
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.var_name: Optional[str] = None
        self.var_idx: Optional[int] = None  # column index in data
        self.exponent: int = 1
        self.const_val: Optional[int] = None
        self.unit_sig: Optional[UnitSig] = None
        self._hash_cache: Optional[int] = None

    @staticmethod
    def variable(name: str, idx: int, exp: int, unit: UnitSig) -> Node:
        n = Node()
        n.var_name = name
        n.var_idx = idx
        n.exponent = exp
        n.unit_sig = pow_sig(unit, exp)
        return n

    @staticmethod
    def constant(val: int) -> Node:
        n = Node()
        n.const_val = val
        n.unit_sig = DIMENSIONLESS
        return n

    @staticmethod
    def operator(op: Op, left: Node, right: Node) -> Node:
        n = Node()
        n.op = op
        n.left = left
        n.right = right
        # Compute unit signature
        if op == Op.MUL:
            n.unit_sig = mul_sig(left.unit_sig, right.unit_sig)
        elif op == Op.DIV:
            n.unit_sig = div_sig(left.unit_sig, right.unit_sig)
        elif op in (Op.ADD, Op.SUB):
            # Addition/subtraction require commensurate signatures
            assert commensurate(left.unit_sig, right.unit_sig), (
                f"Cannot {op.value}: {left.unit_sig} vs {right.unit_sig}"
            )
            n.unit_sig = left.unit_sig
        return n

    def is_leaf(self) -> bool:
        return self.op is None

    def is_variable(self) -> bool:
        return self.var_name is not None

    def is_constant(self) -> bool:
        return self.const_val is not None

    def is_operator(self) -> bool:
        return self.op is not None

    def size(self) -> int:
        """Count total nodes in this subtree."""
        if self.is_leaf():
            return 1
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        """Maximum depth of this subtree."""
        if self.is_leaf():
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    def evaluate(self, row: np.ndarray) -> float:
        """Evaluate expression on a single data row."""
        if self.is_variable():
            val = row[self.var_idx]
            if self.exponent == 1:
                return val
            return val**self.exponent
        if self.is_constant():
            return float(self.const_val)
        # Operator
        lv = self.left.evaluate(row)
        rv = self.right.evaluate(row)
        if self.op == Op.ADD:
            return lv + rv
        elif self.op == Op.SUB:
            return lv - rv
        elif self.op == Op.MUL:
            return lv * rv
        elif self.op == Op.DIV:
            if abs(rv) < 1e-30:
                return float("inf")
            return lv / rv

    def evaluate_batch(self, data: np.ndarray) -> np.ndarray:
        """Evaluate expression on all rows. data shape: (n_samples, n_vars)."""
        if self.is_variable():
            col = data[:, self.var_idx]
            if self.exponent == 1:
                return col.copy()
            with np.errstate(all="ignore"):
                return np.power(col, self.exponent)
        if self.is_constant():
            return np.full(data.shape[0], float(self.const_val))
        lv = self.left.evaluate_batch(data)
        rv = self.right.evaluate_batch(data)
        with np.errstate(all="ignore"):
            if self.op == Op.ADD:
                return lv + rv
            elif self.op == Op.SUB:
                return lv - rv
            elif self.op == Op.MUL:
                return lv * rv
            elif self.op == Op.DIV:
                return np.where(np.abs(rv) < 1e-30, np.inf, lv / rv)

    def to_str(self) -> str:
        """Pretty-print the expression."""
        if self.is_variable():
            if self.exponent == 1:
                return self.var_name
            return f"{self.var_name}^{self.exponent}"
        if self.is_constant():
            return str(self.const_val)
        ls = self.left.to_str()
        rs = self.right.to_str()
        if self.op in (Op.ADD, Op.SUB):
            return f"({ls} {self.op.value} {rs})"
        return f"({ls} {self.op.value} {rs})"

    def __repr__(self) -> str:
        return self.to_str()

    def tree_hash(self) -> int:
        if self._hash_cache is not None:
            return self._hash_cache
        if self.is_variable():
            h = hash(("var", self.var_name, self.exponent))
        elif self.is_constant():
            h = hash(("const", self.const_val))
        else:
            h = hash(("op", self.op, self.left.tree_hash(), self.right.tree_hash()))
        self._hash_cache = h
        return h

    def invalidate_hash(self) -> None:
        """Clear cached hashes up the tree."""
        self._hash_cache = None

    def copy(self) -> Node:
        """Deep copy of this subtree."""
        n = Node()
        n.op = self.op
        if self.left is not None:
            n.left = self.left.copy()
        if self.right is not None:
            n.right = self.right.copy()
        n.var_name = self.var_name
        n.var_idx = self.var_idx
        n.exponent = self.exponent
        n.const_val = self.const_val
        n.unit_sig = self.unit_sig
        n._hash_cache = self._hash_cache
        return n

    def all_subtrees(self) -> list[Node]:
        """Return all subtrees (including self) in pre-order."""
        result = [self]
        if self.is_operator():
            result.extend(self.left.all_subtrees())
            result.extend(self.right.all_subtrees())
        return result

    def recompute_units(self) -> None:
        """Bottom-up recomputation of unit signatures."""
        if self.is_leaf():
            return
        self.left.recompute_units()
        self.right.recompute_units()
        if self.op == Op.MUL:
            self.unit_sig = mul_sig(self.left.unit_sig, self.right.unit_sig)
        elif self.op == Op.DIV:
            self.unit_sig = div_sig(self.left.unit_sig, self.right.unit_sig)
        elif self.op in (Op.ADD, Op.SUB):
            self.unit_sig = self.left.unit_sig


def compute_mse(tree: Node, data: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean squared error of tree predictions vs targets."""
    try:
        preds = tree.evaluate_batch(data)
        if not np.all(np.isfinite(preds)):
            return float("inf")
        return float(np.mean((preds - targets) ** 2))
    except (OverflowError, FloatingPointError, ZeroDivisionError):
        return float("inf")


def compute_mse_linear_scaling(
    tree: Node, data: np.ndarray, targets: np.ndarray
) -> tuple[float, float, float]:
    """Linear scaling: evaluate as (a + b * T). Returns (scaled_mse, a, b)."""
    try:
        preds = tree.evaluate_batch(data)
        if not np.all(np.isfinite(preds)):
            return float("inf"), 0.0, 1.0
        # OLS: minimize sum (targets - a - b*preds)^2
        mean_p = np.mean(preds)
        mean_t = np.mean(targets)
        cov_pt = np.mean(preds * targets) - mean_p * mean_t
        var_p = np.mean(preds**2) - mean_p**2
        if abs(var_p) < 1e-30:
            b = 0.0
            a = mean_t
        else:
            b = cov_pt / var_p
            a = mean_t - b * mean_p
        scaled_preds = a + b * preds
        mse = float(np.mean((scaled_preds - targets) ** 2))
        return mse, a, b
    except (OverflowError, FloatingPointError, ZeroDivisionError):
        return float("inf"), 0.0, 1.0
