"""Graph metrics for LON analysis (Table 5)."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass
class LONMetrics:
    """Graph metrics for a Local Optima Network."""

    nv: int  # number of vertices
    ne: int  # number of edges
    C: float  # clustering coefficient
    Cr: float  # clustering coefficient of random graph
    avg_path_len: float  # average shortest path length (-1 if disconnected)
    pi: int  # connectivity (1 if fully connected, 0 if disconnected)
    S: int  # number of connected components
    nhits: int  # hits to global optimum


def compute_metrics(graph: nx.Graph, nhits: int) -> LONMetrics:
    # Work on undirected view (handles both Graph and DiGraph input)
    G = graph.to_undirected() if graph.is_directed() else graph

    nv = G.number_of_nodes()
    ne = G.number_of_edges()

    if nv <= 1:
        return LONMetrics(
            nv=nv,
            ne=ne,
            C=0.0,
            Cr=0.0,
            avg_path_len=0.0 if nv == 1 else -1.0,
            pi=1 if nv == 1 else 0,
            S=1 if nv >= 1 else 0,
            nhits=nhits,
        )

    # --- Clustering coefficient ---
    C = nx.average_clustering(G)

    # Clustering of equivalent random graph: Cr = p = 2*ne / (nv*(nv-1))
    p = 2.0 * ne / (nv * (nv - 1))
    Cr = p

    # --- Average shortest path length ---
    if nx.is_connected(G):
        avg_path_len = nx.average_shortest_path_length(G)
    else:
        avg_path_len = -1.0  # disconnected

    # --- Connected components ---
    components = list(nx.connected_components(G))
    S = len(components)

    # Connectivity (paper §4.2: 1 if fully connected / S == 1, 0 if disconnected)
    pi = 1 if S == 1 else 0

    return LONMetrics(
        nv=nv,
        ne=ne,
        C=round(C, 2),
        Cr=round(Cr, 2),
        avg_path_len=round(avg_path_len, 2) if avg_path_len >= 0 else avg_path_len,
        pi=pi,
        S=S,
        nhits=nhits,
    )


def format_metrics_table(
    results: dict[str, tuple[LONMetrics, LONMetrics]],
) -> str:
    """
    Format metrics as a table similar to Table 5.

    Parameters:
        results: dict of equation_id -> (no_scaling_metrics, linear_scaling_metrics)

    Returns:
        Formatted string table
    """
    header = (
        f"{'equation':<12} | "
        f"{'nv':>4} {'ne':>4} {'l':>6} {'π':>4} {'S':>3} {'nhits':>5} | "
        f"{'nv':>4} {'ne':>4} {'l':>6} {'π':>4} {'S':>3} {'nhits':>5}"
    )
    sep_line = "-" * len(header)
    label = f"{'':>12} | {'--- no-scaling ---':^30} | {'--- linear-scaling ---':^30}"

    lines = [label, header, sep_line]

    for eq_id, (ns, ls) in results.items():
        line = (
            f"{eq_id:<12} | "
            f"{ns.nv:>4} {ns.ne:>4} "
            f"{ns.avg_path_len:>6.2f} {ns.pi:>4} {ns.S:>3} {ns.nhits:>5} | "
            f"{ls.nv:>4} {ls.ne:>4} "
            f"{ls.avg_path_len:>6.2f} {ls.pi:>4} {ls.S:>3} {ls.nhits:>5}"
        )
        lines.append(line)

    return "\n".join(lines)
