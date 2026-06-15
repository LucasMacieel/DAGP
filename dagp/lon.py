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
    graph: nx.Graph
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
    hash_to_tree: dict[int, Node] = {}

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
            hash_to_tree=hash_to_tree,
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
        search_results,
        hash_to_tree,
    )

    # Find global optimum
    if local_optima_mse:
        global_hash = min(local_optima_mse, key=lambda k: local_optima_mse[k])
        global_mse = local_optima_mse[global_hash]
    else:
        global_hash = 0
        global_mse = np.inf

    # Count hits: number of *distinct* local optima with MSE < 1e-9
    # (paper §4.2, Table 5: "nhits is the number of nodes which represent a hit")
    n_hits = sum(1 for mse in local_optima_mse.values() if mse < 1e-9)

    # Compute evaluations to first hit (Table 4, paper §4.1)
    total_evaluations = sum(r.n_evaluations for r in search_results)
    evaluations_to_hit = -1  # -1 means no hit found
    cumulative_evals = 0
    for r in search_results:
        if r.evals_to_first_hit is not None:
            evaluations_to_hit = cumulative_evals + r.evals_to_first_hit
            break
        cumulative_evals += r.n_evaluations

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


def _run_local_search_worker(
    nb: Node,
    data: np.ndarray,
    targets: np.ndarray,
    var_names: list[str],
    var_units: list,
    use_linear_scaling: bool,
    eval_cache: dict[int, float],
    step_cache: dict[int, Node | None],
) -> tuple[int, int]:
    from dagp.local_search import greedy_local_search
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
    return nb.tree_hash(), result.final_tree.tree_hash()


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
    search_results: list[LocalSearchResult],
    hash_to_tree: dict[int, Node],
) -> None:
    from dagp.operators import generate_all_neighbours
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # 1. Build node_to_optimum mapping and ensure all final optima are cached in hash_to_tree
    node_to_optimum = {}
    for r in search_results:
        final_hash = r.final_tree.tree_hash()
        node_to_optimum[final_hash] = final_hash
        if final_hash not in hash_to_tree:
            hash_to_tree[final_hash] = r.final_tree.copy()
        for h in r.trajectory:
            node_to_optimum[h] = final_hash

    # 2. Group all visited node hashes into sets belonging to each basin
    basin_nodes = {opt_hash: set() for opt_hash in local_optima}
    for h, opt_hash in node_to_optimum.items():
        if opt_hash in basin_nodes:
            basin_nodes[opt_hash].add(h)

    # 3. Connect basins under neighborhood operator (paper §2.2)
    optima_list = list(local_optima.items())
    
    # First pass: Gather all unique unvisited neighbours that require resolution
    unvisited_nodes = {}
    for opt_hash, opt_tree in optima_list:
        nodes_in_basin = basin_nodes.get(opt_hash, set())
        nodes_in_basin.add(opt_hash)

        for h in nodes_in_basin:
            tree_obj = hash_to_tree.get(h)
            if tree_obj is None:
                continue

            neighbours = generate_all_neighbours(tree_obj, var_names, var_units)
            for nb in neighbours:
                nb_hash = nb.tree_hash()
                if nb_hash not in node_to_optimum and nb_hash not in unvisited_nodes:
                    unvisited_nodes[nb_hash] = nb

    # Second pass: Resolve unvisited neighbours in parallel
    if unvisited_nodes:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    _run_local_search_worker,
                    nb,
                    data,
                    targets,
                    var_names,
                    var_units,
                    use_linear_scaling,
                    eval_cache,
                    step_cache,
                )
                for nb in unvisited_nodes.values()
            ]

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="  Resolving unvisited neighbours",
                unit="sol",
            ):
                try:
                    nb_hash, dest_hash = f.result()
                    node_to_optimum[nb_hash] = dest_hash
                except Exception:
                    pass

    # Third pass: Build LON edges sequentially using the resolved node_to_optimum mapping
    for opt_hash, opt_tree in tqdm(
        optima_list, desc="  Building LON edges", unit="optimum"
    ):
        nodes_in_basin = basin_nodes.get(opt_hash, set())
        nodes_in_basin.add(opt_hash)

        for h in nodes_in_basin:
            tree_obj = hash_to_tree.get(h)
            if tree_obj is None:
                continue

            neighbours = generate_all_neighbours(tree_obj, var_names, var_units)
            for nb in neighbours:
                nb_hash = nb.tree_hash()
                dest_hash = node_to_optimum.get(nb_hash)

                # If this neighbour resolves to a different known optimum, create an undirected edge
                if dest_hash in local_optima and dest_hash != opt_hash:
                    if graph.has_edge(opt_hash, dest_hash):
                        graph[opt_hash][dest_hash]["weight"] += 1
                    else:
                        graph.add_edge(opt_hash, dest_hash, weight=1)
