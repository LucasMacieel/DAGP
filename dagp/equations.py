"""Feynman equations metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
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
        raise KeyError(f"Unknown variable '{name}' — add it to dagp/units.csv")
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

# I.12.2: F = q1 * q2 / (4 * pi * epsilon * r**2)
_reg(
    FeynmanEquation(
        id="I.12.2",
        formula="q1 * q2 / (4 * pi * epsilon * r**2)",
        var_names=["q1", "q2", "epsilon", "r"],
        target_unit=_get_unit("F"),
        func=lambda q1, q2, epsilon, r: q1 * q2 / (4 * math.pi * epsilon * r**2),
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

# I.29.4: k = omega / c
_reg(
    FeynmanEquation(
        id="I.29.4",
        formula="omega / c",
        var_names=["omega", "c"],
        target_unit=_get_unit("k"),
        func=lambda omega, c: omega / c,
    )
)

# I.34.8: omega = q * v * B / p
_reg(
    FeynmanEquation(
        id="I.34.8",
        formula="q * v * B / p",
        var_names=["q", "v", "B", "p"],
        target_unit=_get_unit("omega"),
        func=lambda q, v, B, p: q * v * B / p,
    )
)

# I.39.1: En = 1.5 * pr * V
_reg(
    FeynmanEquation(
        id="I.39.1",
        formula="1.5 * pr * V",
        var_names=["pr", "V"],
        target_unit=_get_unit("E_n"),
        func=lambda pr, V: 1.5 * pr * V,
    )
)

# I.25.13: Ve = q / C
_reg(
    FeynmanEquation(
        id="I.25.13",
        formula="q / C",
        var_names=["q", "C"],
        target_unit=_get_unit("Volt"),
        func=lambda q, C: q / C,
    )
)

# I.43.16: v = mob * q * Volt / d
_reg(
    FeynmanEquation(
        id="I.43.16",
        formula="mob * q * Volt / d",
        var_names=["mob", "q", "Volt", "d"],
        target_unit=_get_unit("v"),
        func=lambda mob, q, Volt, d: mob * q * Volt / d,
    )
)

# I.43.31: D = mob * kb * T
_reg(
    FeynmanEquation(
        id="I.43.31",
        formula="mob * kb * T",
        var_names=["mob", "kb", "T"],
        target_unit=_get_unit("D"),
        func=lambda mob, kb, T: mob * kb * T,
    )
)

# II.8.31: Eden = 0.5 * epsilon * Ef**2
_reg(
    FeynmanEquation(
        id="II.8.31",
        formula="0.5 * epsilon * Ef**2",
        var_names=["epsilon", "Ef"],
        target_unit=_get_unit("E_den"),
        func=lambda epsilon, Ef: 0.5 * epsilon * Ef**2,
    )
)

# II.34.2: muM = 0.5 * q * v * r
_reg(
    FeynmanEquation(
        id="II.34.2",
        formula="0.5 * q * v * r",
        var_names=["q", "v", "r"],
        target_unit=_get_unit("mom"),
        func=lambda q, v, r: 0.5 * q * v * r,
    )
)

# III.15.14: m = hbar**2 / (2 * E_n * d**2)
_reg(
    FeynmanEquation(
        id="III.15.14",
        formula="hbar**2 / (2 * E_n * d**2)",
        var_names=["hbar", "E_n", "d"],
        target_unit=_get_unit("m"),
        func=lambda hbar, E_n, d: hbar**2 / (2 * E_n * d**2),
    )
)

EXPERIMENTS = [
    "I.12.2",
    "I.12.5",
    "I.12.1",
    "I.14.3",
    "I.24.6",
    "I.29.4",
    "I.34.8",
    "I.39.1",
    "I.25.13",
    "I.43.16",
    "I.43.31",
    "II.8.31",
    "II.34.2",
    "III.15.14",
]
