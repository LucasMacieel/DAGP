"""Unit signature arithmetic (Table 2)."""

from __future__ import annotations
from typing import NamedTuple


class UnitSig(NamedTuple):
    """Physical unit signature: (m, s, kg, K, V)."""

    m: int = 0
    s: int = 0
    kg: int = 0
    K: int = 0
    V: int = 0


DIMENSIONLESS = UnitSig(0, 0, 0, 0, 0)


def mul_sig(a: UnitSig, b: UnitSig) -> UnitSig:
    return UnitSig(a.m + b.m, a.s + b.s, a.kg + b.kg, a.K + b.K, a.V + b.V)


def div_sig(a: UnitSig, b: UnitSig) -> UnitSig:
    return UnitSig(a.m - b.m, a.s - b.s, a.kg - b.kg, a.K - b.K, a.V - b.V)


def pow_sig(a: UnitSig, n: int) -> UnitSig:
    return UnitSig(a.m * n, a.s * n, a.kg * n, a.K * n, a.V * n)


def commensurate(a: UnitSig, b: UnitSig) -> bool:
    return a == b


def is_dimensionless(sig: UnitSig) -> bool:
    return sig == DIMENSIONLESS
