"""Greedy best-improvement local search (Algorithm 1, §3.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dagp.expression import Node, compute_mse, compute_mse_linear_scaling
from dagp.operators import generate_all_neighbours


@dataclass
class LocalSearchResult:
    """Result of a single local search run."""

    initial_tree: Node
    final_tree: Node
    initial_mse: float
    final_mse: float
    steps: int  # number of improvement steps taken
    n_evaluations: int  # number of fitness evaluations performed
    trajectory: list[int]  # tree hashes visited (for LON edges)


def greedy_local_search(
    initial: Node,
    data: np.ndarray,
    targets: np.ndarray,
    var_names: list[str],
    var_units: list,
    use_linear_scaling: bool = False,
    eval_cache: dict[int, float] | None = None,
    step_cache: dict[int, Node | None] | None = None,
    hash_to_tree: dict[int, Node] | None = None,
) -> LocalSearchResult:

    s = initial.copy()
    n_evals = 0

    current_hash = s.tree_hash()
    if eval_cache is not None and current_hash in eval_cache:
        current_mse = eval_cache[current_hash]
    else:
        if use_linear_scaling:
            current_mse, _, _ = compute_mse_linear_scaling(s, data, targets)
        else:
            current_mse = compute_mse(s, data, targets)
        if eval_cache is not None:
            eval_cache[current_hash] = current_mse

    n_evals += 1

    initial_mse = current_mse
    trajectory = [current_hash]
    if hash_to_tree is not None:
        hash_to_tree[current_hash] = s.copy()
    steps = 0

    if current_mse < 1e-9:
        return LocalSearchResult(
            initial_tree=initial,
            final_tree=s,
            initial_mse=initial_mse,
            final_mse=current_mse,
            steps=steps,
            n_evaluations=n_evals,
            trajectory=trajectory,
        )

    while True:
        current_hash = s.tree_hash()

        # 1. Check if we already know the optimal path from this node
        if step_cache is not None and current_hash in step_cache:
            best_neighbour = step_cache[current_hash]
            if best_neighbour is None:
                break  # previously determined to be a local optimum
            
            s = best_neighbour.copy()
            best_hash = s.tree_hash()
            current_mse = eval_cache[best_hash]
            trajectory.append(best_hash)
            if hash_to_tree is not None:
                hash_to_tree[best_hash] = s.copy()
            steps += 1
            continue

        # Generate full neighbourhood
        neighbours = generate_all_neighbours(s, var_names, var_units)

        # Find the best neighbour
        best_neighbour = None
        best_mse = current_mse

        for nb in neighbours:
            nb_hash = nb.tree_hash()
            if eval_cache is not None and nb_hash in eval_cache:
                nb_mse = eval_cache[nb_hash]
            else:
                if use_linear_scaling:
                    nb_mse, _, _ = compute_mse_linear_scaling(nb, data, targets)
                else:
                    nb_mse = compute_mse(nb, data, targets)
                if eval_cache is not None:
                    eval_cache[nb_hash] = nb_mse

            n_evals += 1

            if nb_mse < 1e-9:
                if hash_to_tree is not None:
                    hash_to_tree[nb_hash] = nb.copy()
                return LocalSearchResult(
                    initial_tree=initial,
                    final_tree=nb,
                    initial_mse=initial_mse,
                    final_mse=nb_mse,
                    steps=steps + 1,
                    n_evaluations=n_evals,
                    trajectory=trajectory + [nb_hash],
                )

            if nb_mse < best_mse:
                best_mse = nb_mse
                best_neighbour = nb

        # Populate step cache before breaking
        if step_cache is not None:
            step_cache[current_hash] = best_neighbour

        # Check for strict improvement
        if best_neighbour is None or best_mse >= current_mse:
            if step_cache is not None:
                step_cache[current_hash] = None
            break  # No improvement, we're at a local optimum

        s = best_neighbour
        current_mse = best_mse
        trajectory.append(s.tree_hash())
        if hash_to_tree is not None:
            hash_to_tree[s.tree_hash()] = s.copy()
        steps += 1

    res = LocalSearchResult(
        initial_tree=initial,
        final_tree=s,
        initial_mse=initial_mse,
        final_mse=current_mse,
        steps=steps,
        n_evaluations=n_evals,
        trajectory=trajectory,
    )

    return res
