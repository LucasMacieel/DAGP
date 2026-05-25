# DAGP — Dimensionally-Aware Genetic Programming

Implementation of experiments from:

> Durasevic et al., *Fitness Landscape Analysis of Dimensionally-Aware Genetic Programming Featuring Feynman Equations* ([arXiv:2004.12762](https://arxiv.org/abs/2004.12762))

## Overview

A dimensionally-constrained greedy local search applied to 27 Feynman physics equations. The search uses physical unit constraints to prune the expression space, extracts **Local Optima Networks (LONs)**, and computes graph-theoretic fitness landscape metrics.

## Architecture

```
dagp/
├── units.py           # UnitSig (5-tuple: m,s,kg,K,V) + arithmetic (Table 2)
├── equations.py       # 27 Feynman equations with variable metadata & unit signatures
├── expression.py      # Expression tree: eval, MSE, linear scaling (OLS), hashing
├── initialization.py  # §3.2 — enumerate monomial expressions matching target signature
├── operators.py       # §3.3 — 5 dimension-preserving neighbourhood operators
├── local_search.py    # §3.4 — Algorithm 1: greedy best-improvement local search
├── lon.py             # LON extraction: local search from all inits + edge building
├── metrics.py         # Graph metrics: nv, ne, C, Cr, l, π, S, nhits (Table 5)
├── visualize.py       # LON network plots (Figs 1-3) + violin plots (Fig 4)
├── gp_baseline.py     # Standard GP baseline using DEAP (§3.5, Table 3)
└── units.csv          # Physical unit signatures for all Feynman variables
run_experiments.py     # Main entry point
```

### Unit System

Unit signatures are 5-tuples `(m, s, kg, K, V)` representing exponents of base SI dimensions. Arithmetic follows Table 2:

- **Multiplication**: exponents add → `[v+v', w+w', ...]`
- **Division**: exponents subtract → `[v-v', w-w', ...]`
- **Addition/Subtraction**: requires commensurate (equal) signatures

### Initialization (§3.2)

Enumerates all monomial expressions `x1^a1 * x2^a2 * ... * xp^ap` where `a_i ∈ [-3, 3]`, keeping only combinations whose resulting unit signature matches the target.

### Neighbourhood Operators (§3.3)

Five dimension-preserving operators applied to every subtree:

1. **Replacement** — swap subtree with one of matching signature
2. **Multiply by integer** — wrap with `(* k)`, `k ∈ [-3,3]\{0,1}`
3. **Divide by integer** — wrap with `(/ k)`
4. **Add commensurate** — wrap with `(+ q)` where `sig(q) == sig(subtree)`
5. **Subtract commensurate** — wrap with `(- q)`

### Local Search (§3.4, Algorithm 1)

Deterministic best-improvement search. Evaluates entire neighbourhood, accepts the single best strict improvement. Fitness = MSE (with optional linear scaling `a + b*T` via OLS). Max tree size: 42 nodes.

### LON Extraction

1. Run local search from **every** initial solution → discover local optima
2. For each local optimum, generate its neighbourhood, run local search → build directed edges between optima

### Graph Metrics (Table 5)

| Metric | Description |
|--------|-------------|
| `nv` | Number of vertices (local optima) |
| `ne` | Number of edges |
| `C` | Global clustering coefficient |
| `Cr` | Clustering of equivalent random graph |
| `l` | Average shortest path length (-1 if disconnected) |
| `π` | Fraction in largest connected component |
| `S` | Number of connected components |
| `nhits` | Initial solutions reaching the global optimum |

## Usage

```bash
uv venv && source .venv/bin/activate
uv pip install -e .

# Run 5-equation subset (default)
python run_experiments.py

# Run all 27 equations
python run_experiments.py --all

# With GP baseline
python run_experiments.py --gp

# Custom sample size
python run_experiments.py --n-samples 200
```

Results are saved to `results/` (LON plots, violin plots, Table 5 metrics).

## Data

Uses the [Feynman Symbolic Regression Database](https://space.mit.edu/home/tegmark/aifeynman.html). Place `Feynman_with_units/` directory in the project root.
