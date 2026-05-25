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
    pi: float  # fraction in largest connected component
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
            pi=1.0 if nv == 1 else 0.0,
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

    # Fraction in largest connected component
    if components:
        largest_cc = max(len(c) for c in components)
        pi = largest_cc / nv
    else:
        pi = 0.0

    return LONMetrics(
        nv=nv,
        ne=ne,
        C=round(C, 2),
        Cr=round(Cr, 2),
        avg_path_len=round(avg_path_len, 2) if avg_path_len >= 0 else avg_path_len,
        pi=round(pi, 2),
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
        f"{'nv':>4} {'ne':>4} {'C':>5} {'Cr':>5} {'l':>6} {'π':>4} {'S':>3} {'nhits':>5} | "
        f"{'nv':>4} {'ne':>4} {'C':>5} {'Cr':>5} {'l':>6} {'π':>4} {'S':>3} {'nhits':>5}"
    )
    sep_line = "-" * len(header)
    label = f"{'':>12} | {'--- no-scaling ---':^42} | {'--- linear-scaling ---':^42}"

    lines = [label, header, sep_line]

    for eq_id, (ns, ls) in results.items():
        line = (
            f"{eq_id:<12} | "
            f"{ns.nv:>4} {ns.ne:>4} {ns.C:>5.2f} {ns.Cr:>5.2f} "
            f"{ns.avg_path_len:>6.2f} {ns.pi:>4.2f} {ns.S:>3} {ns.nhits:>5} | "
            f"{ls.nv:>4} {ls.ne:>4} {ls.C:>5.2f} {ls.Cr:>5.2f} "
            f"{ls.avg_path_len:>6.2f} {ls.pi:>4.2f} {ls.S:>3} {ls.nhits:>5}"
        )
        lines.append(line)

    return "\n".join(lines)
