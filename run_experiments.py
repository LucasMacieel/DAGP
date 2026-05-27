#!/usr/bin/env python3
"""Main experiment runner for DAGP Fitness Landscape Analysis."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from dagp.equations import EQUATIONS, EXPERIMENTS, FeynmanEquation
from dagp.initialization import count_valid_initializations
from dagp.lon import extract_lon, LONResult
from dagp.metrics import compute_metrics, LONMetrics, format_metrics_table
from dagp.visualize import plot_lon_comparison, plot_violin_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data(
    equation_id: str,
    data_dir: str = "Feynman_with_units",
    n_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    filepath = Path(data_dir) / equation_id
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Load space-separated data (no header)
    # Columns: var1, var2, ..., varN, output
    logger.info(f"Loading data from {filepath} (sampling {n_samples} rows)...")

    # Read a subset efficiently
    # The files have 1M rows, we only need a small sample
    raw = np.loadtxt(filepath, max_rows=n_samples * 10)
    if len(raw) > n_samples:
        indices = np.random.RandomState(42).choice(len(raw), n_samples, replace=False)
        raw = raw[indices]

    data = raw[:, :-1]  # all columns except last = inputs
    targets = raw[:, -1]  # last column = output

    return data, targets


def run_dagp_experiment(
    equation: FeynmanEquation,
    data: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
) -> tuple[LONMetrics, LONMetrics, LONResult, LONResult]:
    """
    Run DAGP experiment for a single equation (both scaling variants).

    Returns (no_scaling_metrics, linear_scaling_metrics, lon_ns, lon_ls)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  DAGP Experiment: {equation.id}")
    logger.info(f"  Formula: {equation.formula}")
    logger.info(f"  Variables: {equation.var_names} ({equation.n_vars})")
    logger.info(f"{'=' * 60}")

    # Check initialization count
    n_init = count_valid_initializations(equation.var_units, equation.target_unit)
    logger.info(f"  Valid initial solutions: {n_init}")

    if n_init == 0:
        logger.warning("  No valid initial solutions! Skipping.")
        empty = LONMetrics(0, 0, 0.0, 0.0, -1.0, 0.0, 0, 0)
        return empty, empty, None, None

    if n_init > 5000:
        logger.warning(f"  Too many initial solutions ({n_init}). This may be slow.")

    # --- No scaling ---
    logger.info("\n  --- No scaling ---")
    t0 = time.time()
    lon_ns = extract_lon(
        equation,
        data,
        targets,
        use_linear_scaling=False,
    )
    t_ns = time.time() - t0
    logger.info(f"  No-scaling completed in {t_ns:.1f}s")

    metrics_ns = compute_metrics(lon_ns.graph, lon_ns.n_hits)
    evals_ns = lon_ns.evaluations_to_hit
    evals_ns_str = str(evals_ns) if evals_ns >= 0 else "-"
    logger.info(
        f"  Metrics: nv={metrics_ns.nv}, ne={metrics_ns.ne}, C={metrics_ns.C}, Cr={metrics_ns.Cr}, l={metrics_ns.avg_path_len}, nhits={metrics_ns.nhits}, "
        f"pi={metrics_ns.pi}, S={metrics_ns.S}, nhits={metrics_ns.nhits}, evals_to_hit={evals_ns_str}"
    )

    # --- Linear scaling ---
    logger.info("\n  --- Linear scaling ---")
    t0 = time.time()
    lon_ls = extract_lon(
        equation,
        data,
        targets,
        use_linear_scaling=True,
    )
    t_ls = time.time() - t0
    logger.info(f"  Linear-scaling completed in {t_ls:.1f}s")

    metrics_ls = compute_metrics(lon_ls.graph, lon_ls.n_hits)
    evals_ls = lon_ls.evaluations_to_hit
    evals_ls_str = str(evals_ls) if evals_ls >= 0 else "-"
    logger.info(
        f"  Metrics: nv={metrics_ls.nv}, ne={metrics_ls.ne}, C={metrics_ls.C}, Cr={metrics_ls.Cr}, l={metrics_ls.avg_path_len}, nhits={metrics_ls.nhits}, "
        f"pi={metrics_ls.pi}, S={metrics_ls.S}, nhits={metrics_ls.nhits}, evals_to_hit={evals_ls_str}"
    )

    # --- Plot LON comparison ---
    plot_path = output_dir / f"lon_{equation.id.replace('.', '_')}.png"
    try:
        plot_lon_comparison(lon_ns, lon_ls, equation.id, save_path=str(plot_path))
        logger.info(f"  LON plot saved: {plot_path}")
    except Exception as e:
        logger.warning(f"  Failed to save LON plot: {e}")

    return metrics_ns, metrics_ls, lon_ns, lon_ls


def run_gp_experiment(
    equation: FeynmanEquation,
    data: np.ndarray,
    targets: np.ndarray,
    use_linear_scaling: bool = True,
):
    """Run GP baseline for a single equation."""
    from dagp.gp_baseline import run_gp_baseline

    scaling_str = "scaling" if use_linear_scaling else "no scaling"
    logger.info(f"\n  --- GP Baseline ({scaling_str}) ---")
    t0 = time.time()

    result = run_gp_baseline(
        equation_id=equation.id,
        data=data,
        targets=targets,
        var_names=equation.var_names,
        n_runs=50,
        use_linear_scaling=use_linear_scaling,
    )

    t_gp = time.time() - t0
    logger.info(f"  GP ({scaling_str}) completed in {t_gp:.1f}s")
    logger.info(f"  Best MSE: {result.best_mse:.6e}")
    logger.info(f"  Median MSE: {result.median_mse:.6e}")
    logger.info(f"  Hits: {result.n_hits}/50")
    if result.avg_evals_to_hit is not None:
        logger.info(f"  Avg evals to hit: {result.avg_evals_to_hit:.0f}")
    logger.info(f"  Best expr: {result.best_expression}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="DAGP Fitness Landscape Analysis experiments"
    )
    parser.add_argument(
        "--gp",
        action="store_true",
        help="Also run GP baseline (slower)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Feynman_with_units",
        help="Directory containing Feynman data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of data samples to use (default: 100, per paper §4)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DAGP Experiment Runner")
    logger.info(f"  Equations: {EXPERIMENTS}")
    logger.info(f"  Data dir: {args.data_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Samples: {args.n_samples}")
    logger.info(f"  GP baseline: {args.gp}")

    # Run experiments
    all_metrics: dict[str, tuple[LONMetrics, LONMetrics]] = {}
    all_lons: dict[str, tuple[LONResult, LONResult]] = {}
    gp_results = {}

    for eq_id in EXPERIMENTS:
        if eq_id not in EQUATIONS:
            logger.warning(f"Equation {eq_id} not defined, skipping.")
            continue

        equation = EQUATIONS[eq_id]

        # Load data
        try:
            data, targets = load_data(
                eq_id, data_dir=args.data_dir, n_samples=args.n_samples
            )
        except FileNotFoundError as e:
            logger.error(f"  {e}")
            continue

        # Verify data dimensions
        if data.shape[1] != equation.n_vars:
            logger.error(
                f"  Data has {data.shape[1]} columns but equation has "
                f"{equation.n_vars} variables. Skipping."
            )
            continue

        # Run DAGP
        metrics_ns, metrics_ls, lon_ns, lon_ls = run_dagp_experiment(
            equation, data, targets, output_dir
        )
        all_metrics[eq_id] = (metrics_ns, metrics_ls)
        if lon_ns is not None:
            all_lons[eq_id] = (lon_ns, lon_ls)

        # Run GP baseline if requested
        if args.gp:
            gp_ns = run_gp_experiment(equation, data, targets, use_linear_scaling=False)
            gp_ls = run_gp_experiment(equation, data, targets, use_linear_scaling=True)
            gp_results[eq_id] = (gp_ns, gp_ls)

    # Print Table 4: evaluations to hit
    if all_lons:
        logger.info(f"\n\n{'=' * 80}")
        logger.info("  EVALUATIONS TABLE (Table 4 reproduction)")
        logger.info(f"{'=' * 80}\n")
        header = f"{'Eq. label':<12}  {'no scaling':>12}  {'scaling':>12}  {'GP no scaling':>15}  {'GP scaling':>15}"
        print(header)
        print("-" * len(header))
        for eq_id, (lon_ns, lon_ls) in all_lons.items():
            ns_str = (
                str(lon_ns.evaluations_to_hit)
                if lon_ns.evaluations_to_hit >= 0
                else "-"
            )
            ls_str = (
                str(lon_ls.evaluations_to_hit)
                if lon_ls.evaluations_to_hit >= 0
                else "-"
            )
            gp_ns_str = "-"
            gp_ls_str = "-"
            if eq_id in gp_results:
                gp_ns, gp_ls = gp_results[eq_id]
                if gp_ns.n_hits > 0 and gp_ns.avg_evals_to_hit is not None:
                    gp_ns_str = f"{gp_ns.avg_evals_to_hit:.0f} ({gp_ns.n_hits})"
                if gp_ls.n_hits > 0 and gp_ls.avg_evals_to_hit is not None:
                    gp_ls_str = f"{gp_ls.avg_evals_to_hit:.0f} ({gp_ls.n_hits})"
            print(
                f"{eq_id:<12}  {ns_str:>12}  {ls_str:>12}  {gp_ns_str:>15}  {gp_ls_str:>15}"
            )
        print()

        # Save evaluations table
        eval_path = output_dir / "table4_evaluations.txt"
        with open(eval_path, "w") as f:
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for eq_id, (lon_ns, lon_ls) in all_lons.items():
                ns_str = (
                    str(lon_ns.evaluations_to_hit)
                    if lon_ns.evaluations_to_hit >= 0
                    else "-"
                )
                ls_str = (
                    str(lon_ls.evaluations_to_hit)
                    if lon_ls.evaluations_to_hit >= 0
                    else "-"
                )
                gp_ns_str = "-"
                gp_ls_str = "-"
                if eq_id in gp_results:
                    gp_ns, gp_ls = gp_results[eq_id]
                    if gp_ns.n_hits > 0 and gp_ns.avg_evals_to_hit is not None:
                        gp_ns_str = f"{gp_ns.avg_evals_to_hit:.0f} ({gp_ns.n_hits})"
                    if gp_ls.n_hits > 0 and gp_ls.avg_evals_to_hit is not None:
                        gp_ls_str = f"{gp_ls.avg_evals_to_hit:.0f} ({gp_ls.n_hits})"
                f.write(
                    f"{eq_id:<12}  {ns_str:>12}  {ls_str:>12}  {gp_ns_str:>15}  {gp_ls_str:>15}\n"
                )
        logger.info(f"Evaluations table saved: {eval_path}")

    # Print Table 5: graph metrics
    if all_metrics:
        logger.info(f"\n\n{'=' * 80}")
        logger.info("  RESULTS TABLE (Table 5 reproduction)")
        logger.info(f"{'=' * 80}\n")
        table = format_metrics_table(all_metrics)
        print(table)

        # Save table
        table_path = output_dir / "table5_metrics.txt"
        with open(table_path, "w") as f:
            f.write(table)
        logger.info(f"\nTable saved: {table_path}")

        # Violin plots
        if len(all_metrics) >= 2:
            violin_path = output_dir / "violin_plots.png"
            try:
                plot_violin_metrics(all_metrics, save_path=str(violin_path))
                logger.info(f"Violin plots saved: {violin_path}")
            except Exception as e:
                logger.warning(f"Failed to save violin plots: {e}")

    # Print GP results
    if gp_results:
        logger.info(f"\n\n{'=' * 80}")
        logger.info("  GP BASELINE RESULTS")
        logger.info(f"{'=' * 80}\n")
        for eq_id, (res_ns, res_ls) in gp_results.items():
            avg_str_ns = (
                f", avg_evals={res_ns.avg_evals_to_hit:.0f}"
                if res_ns.avg_evals_to_hit is not None
                else ""
            )
            avg_str_ls = (
                f", avg_evals={res_ls.avg_evals_to_hit:.0f}"
                if res_ls.avg_evals_to_hit is not None
                else ""
            )
            print(
                f"{eq_id} (no scaling): best={res_ns.best_mse:.4e}, "
                f"median={res_ns.median_mse:.4e}, hits={res_ns.n_hits}/50{avg_str_ns}"
            )
            print(
                f"{eq_id} (scaling):    best={res_ls.best_mse:.4e}, "
                f"median={res_ls.median_mse:.4e}, hits={res_ls.n_hits}/50{avg_str_ls}"
            )


if __name__ == "__main__":
    main()
