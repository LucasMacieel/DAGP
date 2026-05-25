"""Local Optima Network (LON) extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tqdm import tqdm

import networkx as nx
import numpy as np

from dagp.equations import FeynmanEquation
from dagp.expression import Node
from dagp.initialization import generate_initial_solutions
from dagp.local_search import greedy_local_search, LocalSearchResult

logger = logging.getLogger(__name__)


@dataclass
class LONResult:
    """Results of LON extraction for a single equation."""

    equation_id: str
    graph: nx.DiGraph
    local_optima: dict[int, Node]  # hash -> tree
    local_optima_mse: dict[int, float]  # hash -> MSE
    global_optimum_hash: int  # hash of best local optimum
    global_optimum_mse: float
    n_initial_solutions: int
    n_hits: int  # how many initial solutions reach the global optimum
    total_evaluations: int  # total fitness evaluations across all initial solutions
    evaluations_to_hit: int  # cumulative evals to first hit (-1 if no hit)
    search_results: list[LocalSearchResult] = field(repr=False)
    use_linear_scaling: bool = False


def extract_lon(
    equation: FeynmanEquation,
    data: np.ndarray,
    targets: np.ndarray,
    use_linear_scaling: bool = False,
) -> LONResult:
    # 1. Generate initial solutions
    initial_solutions = generate_initial_solutions(
        equation.var_names,
        equation.var_units,
        equation.target_unit,
    )

    n_init = len(initial_solutions)
    logger.info(
        f"[{equation.id}] Generated {n_init} initial solutions "
        f"(scaling={'linear' if use_linear_scaling else 'no'})"
    )

    # 2. Run local search from each initial solution
    graph = nx.Graph()
    local_optima: dict[int, Node] = {}
    local_optima_mse: dict[int, float] = {}
    search_results: list[LocalSearchResult] = []
    
    eval_cache: dict[int, float] = {}
    step_cache: dict[int, Node | None] = {}

    scaling_label = "linear" if use_linear_scaling else "no-scaling"
    for init_sol in tqdm(
        initial_solutions,
        desc=f"[{equation.id}] LS ({scaling_label})",
        unit="sol",
    ):
        result = greedy_local_search(
            initial=init_sol,
            data=data,
            targets=targets,
            var_names=equation.var_names,
            var_units=equation.var_units,
            use_linear_scaling=use_linear_scaling,
            eval_cache=eval_cache,
            step_cache=step_cache,
        )
        search_results.append(result)

        # Record the local optimum (using structural tree hash for identity)
        final_hash = result.final_tree.tree_hash()
        if final_hash not in local_optima:
            local_optima[final_hash] = result.final_tree
            local_optima_mse[final_hash] = result.final_mse

        # Add node to graph
        if not graph.has_node(final_hash):
            graph.add_node(final_hash, mse=result.final_mse)

    _build_lon_edges(
        graph,
        local_optima,
        local_optima_mse,
        data,
        targets,
        equation.var_names,
        equation.var_units,
        use_linear_scaling,
        eval_cache,
        step_cache,
    )

    # Find global optimum
    if local_optima_mse:
        global_hash = min(local_optima_mse, key=local_optima_mse.get)
        global_mse = local_optima_mse[global_hash]
    else:
        global_hash = 0
        global_mse = float("inf")

    # Count hits: number of *distinct* local optima with MSE < 1e-9
    # (paper §4.2, Table 5: "nhits is the number of nodes which represent a hit")
    n_hits = sum(1 for mse in local_optima_mse.values() if mse < 1e-9)

    # Compute evaluations to first hit (Table 4, paper §4.1)
    total_evaluations = sum(r.n_evaluations for r in search_results)
    evaluations_to_hit = -1  # -1 means no hit found
    cumulative_evals = 0
    for r in search_results:
        cumulative_evals += r.n_evaluations
        if r.final_mse < 1e-9:
            evaluations_to_hit = cumulative_evals
            break

    evals_str = str(evaluations_to_hit) if evaluations_to_hit >= 0 else "-"
    logger.info(
        f"[{equation.id}] LON: {len(local_optima)} optima, "
        f"{graph.number_of_edges()} edges, {n_hits} hits, "
        f"evals to hit: {evals_str} (total: {total_evaluations})"
    )

    return LONResult(
        equation_id=equation.id,
        graph=graph,
        local_optima=local_optima,
        local_optima_mse=local_optima_mse,
        global_optimum_hash=global_hash,
        global_optimum_mse=global_mse,
        n_initial_solutions=n_init,
        n_hits=n_hits,
        total_evaluations=total_evaluations,
        evaluations_to_hit=evaluations_to_hit,
        search_results=search_results,
        use_linear_scaling=use_linear_scaling,
    )


def _build_lon_edges(
    graph: nx.Graph,
    local_optima: dict[int, Node],
    local_optima_mse: dict[int, float],
    data: np.ndarray,
    targets: np.ndarray,
    var_names: list[str],
    var_units: list,
    use_linear_scaling: bool,
    eval_cache: dict[int, float],
    step_cache: dict[int, Node | None],
) -> None:
    from dagp.operators import generate_all_neighbours

    optima_list = list(local_optima.items())

    for opt_hash, opt_tree in tqdm(
        optima_list, desc="  Building LON edges", unit="optimum"
    ):
        # Generate neighbourhood of this local optimum
        neighbours = generate_all_neighbours(opt_tree, var_names, var_units)

        for nb in neighbours:
            # Run local search from this neighbour
            result = greedy_local_search(
                initial=nb,
                data=data,
                targets=targets,
                var_names=var_names,
                var_units=var_units,
                use_linear_scaling=use_linear_scaling,
                eval_cache=eval_cache,
                step_cache=step_cache,
            )

            dest_hash = result.final_tree.tree_hash()

            # Only add edges between known local optima (paper §2.2:
            # the vertex set is fixed after initial local search)
            if dest_hash in local_optima and dest_hash != opt_hash:
                if graph.has_edge(opt_hash, dest_hash):
                    graph[opt_hash][dest_hash]["weight"] += 1
                else:
                    graph.add_edge(opt_hash, dest_hash, weight=1)
