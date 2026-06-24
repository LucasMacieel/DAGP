# DAGP — Dimensionally-Aware Genetic Programming

Implementation of experiments and comparison baseline from:

> Durasevic et al., *Fitness Landscape Analysis of Dimensionally-Aware Genetic Programming Featuring Feynman Equations* ([arXiv:2004.12762](https://arxiv.org/abs/2004.12762))

## Overview

A dimensionally-constrained greedy local search applied to physical equations. The search uses physical unit signatures to prune the expression search space, extracts **Local Optima Networks (LONs)** to model the fitness landscape, and computes graph-theoretic metrics. A highly optimized, custom Genetic Programming (GP) baseline is provided as a comparison.

---

## Architecture

The project is structured modularly under the `dagp/` package:

```
dagp/
├── units.py           # UnitSig NamedTuple (m, s, kg, K, V) & commensurate arithmetic (Table 2)
├── equations.py       # Feynman equations registry (15 equations, e.g., I.12.5, III.13.18) with target units
├── expression.py      # Expression tree nodes, batch evaluations, MSE/OLS linear scaling, hashing
├── initialization.py  # §3.2 — Monomial enumeration matching target unit signatures
├── operators.py       # §3.3 — 5 dimension-preserving neighborhood operators
├── local_search.py    # §3.4 — Algorithm 1: greedy best-improvement search with caching
├── lon.py             # Basin-to-basin LON edge construction and extraction
├── metrics.py         # Table 5 graph metrics (clustering, component size, average path length)
├── visualize.py       # Side-by-side LON network comparison and violin plots
├── gp_baseline.py     # Custom steady-state DEAP GP symbolic regression (§3.5, Table 3)
└── units.csv          # Base SI unit database for Feynman variables
run_experiments.py     # Central CLI experiment runner
```

---

## Core Components

### 1. Physical Unit System
Unit signatures are represented as 5-tuples `(m, s, kg, K, V)` representing the integer exponents of base SI dimensions (length, time, mass, temperature, electric voltage). 
Arithmetic rules strictly follow physical dimensional analysis (Table 2 of the paper):
* **Multiplication**: Exponents sum.
* **Division**: Exponents subtract.
* **Addition / Subtraction**: Commensurate constraint (requires identical unit signatures).

### 2. Initialization & Structural Bloat Prevention
Valid initial solutions are generated as monomials $x_1^{a_1} \cdot x_2^{a_2} \cdots x_p^{a_p}$ within an exponent range of $a_i \in [-3, 3]$. 
* **Bloat Prevention**: Any variable with an exponent of `0` is completely filtered out during tree compilation, preventing structural bloat (such as multiplying by $x^0$) and ensuring compact, high-quality starting trees.

### 3. Dimension-Preserving Neighborhood Operators
To navigate the landscape without violating physical unit consistency, five operators are applied to every subtree in the expression:
1. **Replacement**: Swaps a subtree with a commensurate monomial subtree.
2. **Multiply by Integer**: Wraps a subtree with `(* k)` where $k \in [-3, 3] \setminus \{0, 1\}$.
3. **Divide by Integer**: Wraps a subtree with `(/ k)` where $k \in [-3, 3] \setminus \{0, 1\}$.
4. **Add Commensurate**: Wraps a subtree with `(+ q)` where $\text{sig}(q) = \text{sig}(\text{subtree})$.
5. **Subtract Commensurate**: Wraps a subtree with `(- q)` where $\text{sig}(q) = \text{sig}(\text{subtree})$.

*Identity pruning is enforced by excluding $k = 0, 1$ in multipliers/dividers. Expression trees are capped at a maximum size of `MAX_TREE_SIZE = 42` nodes.*

### 4. Greedy Best-Improvement Local Search
Deterministic local search (Algorithm 1) traverses the neighborhood, selecting the single best strict improvement in Mean Squared Error (MSE). Caching of intermediate evaluation outcomes and transition paths is utilized to accelerate execution.
* **Linear Scaling**: Optionally uses Ordinary Least Squares (OLS) linear scaling ($a + b \cdot T$) to scale predictions.

### 5. Local Optima Networks (LON) & Basin Connection Logic
LONs are built by running the greedy local search from **every** valid initial solution.
* **Basin-to-Basin Connection**: Transitions between basins are mapped. If a single-step neighborhood operator applied to any expression in basin $A$ results in a tree that resolves to local optimum $B$ under greedy search, an undirected edge is established between $A$ and $B$, weighted by transition frequencies.

### 6. Graph-Theoretic Metrics

Graph analysis is performed on the undirected view of the network (aligning with Table 5 of the paper). The formatted results table (`table5_metrics.txt`) prints the following columns:
* `nv`: Number of vertices (discovered local optima).
* `ne`: Number of edges.
* `l`: Average shortest path length (returns `-1.0` if the network is disconnected).
* `π` (pi): Connectivity indicator: `1` if the graph has exactly one connected component ($S = 1$), `0` otherwise.
* `S`: Number of connected components.
* `nhits`: Count of distinct local optima qualifying as successful hits ($\text{MSE} < 10^{-9}$).

*(Note: Clustering coefficients `C` and `Cr` are computed during analysis but are omitted from the final printed reports and plots to match the project formatting).*

---

## Genetic Programming (GP) Comparison Baseline

The baseline uses a highly customized steady-state Symbolic Regression engine built on `DEAP`:

* **Steady-State Evolution (MGG)**: A tournament-based evolutionary scheme. Three individuals are randomly chosen; the worst is replaced by the mutated offspring of the top two. Early stopping is triggered once any individual achieves a hit ($\text{MSE} < 10^{-9}$).
* **Parameters**: Population size: `500` | Max fitness evaluations: `100,000` | Max tree depth: `6` | Mutation probability: `0.5`.
* **Crossover Operators**:
  * **Subtree Crossover**: Standard DEAP subtree swapping.
  * **Leaf-Biased Crossover**: Subtree swapping with a 10% terminal selection bias.
  * **Size-Fair Crossover**: Swaps subtrees of similar sizes to curb code bloat.
  * **Context-Preserved Crossover**: Swaps subtrees that occupy identical structural locations.
  * **GP Uniform Crossover (Poli & Langdon)**: Swaps nodes in the common parent region or subtrees on the structural boundaries.
* **Mutation Operators**: Subtree, Hoist, Node Replacement, Permutation, and Shrink.

---

## Usage

Set up the environment with `uv` and install the package in editable mode:

```bash
# Set up virtual environment
uv venv
source .venv/bin/activate

# Install dependencies and the local package
uv pip install -e .
```

### Running Experiments

```bash
# Run the core local search & LON extraction pipeline
python run_experiments.py

# Run the pipeline including the standard GP comparison baseline
python run_experiments.py --gp

# Customize the number of sample data points
python run_experiments.py --n-samples 200 --gp
```

### Options

* `--gp`: Run GP baseline comparison runs (50 independent runs per equation).
* `--n-samples`: Number of data rows sampled from data files (default: `100`).
* `--data-dir`: Custom path to the Feynman database (default: `Feynman_with_units`).
* `--output-dir`: Output directory for files (default: `results`).

### Outputs

All results are output directly to `results/`:
* `table4_evaluations.txt`: Evaluations to hit for both local search and GP.
* `table5_metrics.txt`: Extracted graph metrics for no-scaling and linear-scaling variants.
* `lon_*.png`: Side-by-side graph plots showing the extracted Local Optima Networks.
* `violin_plots.png`: Violin distributions of the landscape metrics.

---

## Data

The pipeline runs on data from the [Feynman Symbolic Regression Database](https://space.mit.edu/home/tegmark/aifeynman.html). Place the `Feynman_with_units/` directory inside the project root before running the script.
