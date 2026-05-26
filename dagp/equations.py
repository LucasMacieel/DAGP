"""Feynman equations metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dagp.units import UnitSig


def _load_unit_table() -> dict[str, UnitSig]:
    """Load variable unit signatures from units.csv."""
    csv_path = Path(__file__).parent / "units.csv"
    table: dict[str, UnitSig] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Variable"].strip()
            if not name:
                continue
            table[name] = UnitSig(
                m=int(row["m"].strip()),
                s=int(row["s"].strip()),
                kg=int(row["kg"].strip()),
                K=int(row["T"].strip()),
                V=int(row["V"].strip()),
            )
    return table


_UNIT_TABLE: dict[str, UnitSig] = _load_unit_table()


def _get_unit(name: str) -> UnitSig:
    """Look up a variable's unit signature."""
    if name not in _UNIT_TABLE:
        raise KeyError(
            f"Unknown variable '{name}' — add it to dagp/units.csv"
        )
    return _UNIT_TABLE[name]


@dataclass
class FeynmanEquation:
    """Metadata for a single Feynman equation."""

    id: str
    formula: str
    var_names: list[str]
    target_unit: UnitSig
    func: Callable  # f(x1, x2, ...) -> y
    var_units: list[UnitSig] = field(init=False)
    n_vars: int = field(init=False)

    def __post_init__(self) -> None:
        self.var_units = [_get_unit(name) for name in self.var_names]
        self.n_vars = len(self.var_names)


# -----------------------------------------------------------------------
#  Equations registry
# -----------------------------------------------------------------------

EQUATIONS: dict[str, FeynmanEquation] = {}


def _reg(eq: FeynmanEquation) -> FeynmanEquation:
    EQUATIONS[eq.id] = eq
    return eq


# I.24.6: E = 1/4 m (omega^2 + omega_0^2) x^2
_reg(
    FeynmanEquation(
        id="I.24.6",
        formula="0.25 * m * (omega**2 + omega_0**2) * x**2",
        var_names=["m", "omega", "omega_0", "x"],
        target_unit=_get_unit("E_n"),  # Energy
        func=lambda m, omega, omega_0, x: 0.25 * m * (omega**2 + omega_0**2) * x**2,
    )
)

# I.12.5: F = q2 * Ef
_reg(
    FeynmanEquation(
        id="I.12.5",
        formula="q2 * Ef",
        var_names=["q2", "Ef"],
        target_unit=_get_unit("F"),  # Force
        func=lambda q2, Ef: q2 * Ef,
    )
)

# I.12.1: F = mu * Nn
_reg(
    FeynmanEquation(
        id="I.12.1",
        formula="mu * Nn",
        var_names=["mu", "Nn"],
        target_unit=_get_unit("F"),
        func=lambda mu, Nn: mu * Nn,
    )
)

# I.14.3: U = m * g * z
_reg(
    FeynmanEquation(
        id="I.14.3",
        formula="m * g * z",
        var_names=["m", "g", "z"],
        target_unit=_get_unit("U"),
        func=lambda m, g, z: m * g * z,
    )
)

# I.14.4: U = 0.5 * k_spring * x**2
_reg(
    FeynmanEquation(
        id="I.14.4",
        formula="0.5 * k_spring * x**2",
        var_names=["k_spring", "x"],
        target_unit=_get_unit("U"),
        func=lambda k_spring, x: 0.5 * k_spring * x**2,
    )
)

# "I.12.5", "I.12.1", "I.14.3", I.14.4, 
EXPERIMENTS = ["I.24.6"]
