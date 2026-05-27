"""Standard GP symbolic regression baseline using DEAP (§3.5, Table 3)."""

from __future__ import annotations

import functools
import math
import random
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from deap import algorithms, base, creator, gp, tools


@dataclass
class GPResult:
    """Result of GP symbolic regression for one equation."""

    equation_id: str
    best_mse: float
    best_expression: str
    median_mse: float
    mean_mse: float
    n_hits: int  # number of runs with MSE < threshold
    avg_evals_to_hit: float | None  # average evaluations for successful runs
    all_mse: list[float]


def _protected_div(a, b):
    """Protected division to avoid ZeroDivisionError."""
    if abs(b) < 1e-10:
        return 1.0
    return a / b


def run_gp_baseline(
    equation_id: str,
    data: np.ndarray,
    targets: np.ndarray,
    var_names: list[str],
    n_runs: int = 50,
    pop_size: int = 500,
    max_evals: int = 100_000,
    max_depth: int = 6,
    mut_prob: float = 0.5,
    cx_prob: float = 0.9,
    hit_threshold: float = 1e-9,
    seed: int = 42,
    use_linear_scaling: bool = True,
) -> GPResult:
    # Clean up any previous DEAP creator definitions
    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual

    # Define primitive set
    pset = gp.PrimitiveSet("MAIN", len(var_names))
    pset.addPrimitive(lambda a, b: a + b, 2, name="add")
    pset.addPrimitive(lambda a, b: a - b, 2, name="sub")
    pset.addPrimitive(lambda a, b: a * b, 2, name="mul")
    pset.addPrimitive(_protected_div, 2, name="div")
    pset.addPrimitive(math.sin, 1, name="sin")
    pset.addPrimitive(math.cos, 1, name="cos")

    # Rename arguments to match variable names
    for i, name in enumerate(var_names):
        pset.renameArguments(**{f"ARG{i}": name})

    # DEAP setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def eval_individual(individual) -> tuple[float]:
        func = toolbox.compile(expr=individual)
        try:
            preds = np.array([func(*row) for row in data])
            if not np.all(np.isfinite(preds)):
                return (float("inf"),)
            if use_linear_scaling:
                # Linear scaling
                mean_p = np.mean(preds)
                mean_t = np.mean(targets)
                var_p = np.var(preds)
                if var_p < 1e-30:
                    return (float(np.mean((targets - mean_t) ** 2)),)
                cov_pt = np.mean(preds * targets) - mean_p * mean_t
                b = cov_pt / var_p
                a = mean_t - b * mean_p
                scaled = a + b * preds
                mse = float(np.mean((scaled - targets) ** 2))
            else:
                # Direct MSE without scaling
                mse = float(np.mean((preds - targets) ** 2))
            return (mse,)
        except (OverflowError, FloatingPointError, ZeroDivisionError, ValueError):
            return (float("inf"),)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Custom crossover operators
    def cxSizeFair(ind1, ind2):
        if len(ind1) < 2 or len(ind2) < 2:
            return gp.cxOnePoint(ind1, ind2)
        idx1 = random.randint(1, len(ind1) - 1)
        slice1 = ind1.searchSubtree(idx1)
        size1 = slice1.stop - slice1.start
        
        best_idx2 = 1
        best_diff = float('inf')
        best_slice2 = ind2.searchSubtree(1)
        for i in range(1, len(ind2)):
            s2 = ind2.searchSubtree(i)
            diff = abs((s2.stop - s2.start) - size1)
            if diff < best_diff:
                best_diff = diff
                best_slice2 = s2
                if diff == 0:
                    break
        slice2 = best_slice2
        new_ind1 = ind1[:slice1.start] + ind2[slice2] + ind1[slice1.stop:]
        new_ind2 = ind2[:slice2.start] + ind1[slice1] + ind2[slice2.stop:]
        return ind1.__class__(new_ind1), ind2.__class__(new_ind2)

    def cxContextPreserved(ind1, ind2):
        def _common_region(idx1, idx2):
            yield idx1, idx2
            node1, node2 = ind1[idx1], ind2[idx2]
            if node1.arity == node2.arity:
                c1, c2 = idx1 + 1, idx2 + 1
                for _ in range(node1.arity):
                    yield from _common_region(c1, c2)
                    c1, c2 = ind1.searchSubtree(c1).stop, ind2.searchSubtree(c2).stop
        common_pairs = list(_common_region(0, 0))
        if len(common_pairs) <= 1:
            return gp.cxOnePoint(ind1, ind2)
        idx1, idx2 = random.choice(common_pairs[1:])
        slice1, slice2 = ind1.searchSubtree(idx1), ind2.searchSubtree(idx2)
        new_ind1 = ind1[:slice1.start] + ind2[slice2] + ind1[slice1.stop:]
        new_ind2 = ind2[:slice2.start] + ind1[slice1] + ind2[slice2.stop:]
        return ind1.__class__(new_ind1), ind2.__class__(new_ind2)

    def cxGPUniform(ind1, ind2, swap_prob=0.5):
        """Standard tree-based uniform crossover following the user's recursive logic."""
        class TempNode:
            def __init__(self, value=None):
                self.value = value
                self.children = []

        def prefix_to_tree(flat_list):
            iterator = iter(flat_list)
            def parse():
                node = next(iterator)
                temp = TempNode(node)
                for _ in range(node.arity):
                    temp.children.append(parse())
                return temp
            return parse()

        def tree_to_prefix(root):
            flat = []
            def serialize(node):
                if node is None:
                    return
                flat.append(node.value)
                for child in node.children:
                    serialize(child)
            serialize(root)
            return flat

        def uniform_crossover(node_a, node_b):
            if node_a is None and node_b is None:
                return None, None

            # Structural mismatch → subtree swap fallback
            if node_a is None or node_b is None:
                return node_b, node_a

            offspring1 = TempNode()
            offspring2 = TempNode()

            if random.random() < swap_prob:
                offspring1.value = node_b.value
                offspring2.value = node_a.value
            else:
                offspring1.value = node_a.value
                offspring2.value = node_b.value

            arity_a = len(node_a.children)
            arity_b = len(node_b.children)
            
            arity1 = offspring1.value.arity
            arity2 = offspring2.value.arity

            for i in range(max(arity1, arity2)):
                child_a = node_a.children[i] if i < arity_a else None
                child_b = node_b.children[i] if i < arity_b else None

                c1, c2 = uniform_crossover(child_a, child_b)
                if i < arity1:
                    offspring1.children.append(c1 if c1 is not None else c2)
                if i < arity2:
                    offspring2.children.append(c2 if c2 is not None else c1)

            return offspring1, offspring2

        # Convert prefix list to recursive tree representation
        tree_a = prefix_to_tree(ind1)
        tree_b = prefix_to_tree(ind2)

        # Perform the uniform crossover
        off1_tree, off2_tree = uniform_crossover(tree_a, tree_b)

        # Convert back to prefix list
        off1_flat = tree_to_prefix(off1_tree)
        off2_flat = tree_to_prefix(off2_tree)

        # Update in-place
        ind1[:] = off1_flat
        ind2[:] = off2_flat
        return ind1, ind2


    def mate_random(ind1, ind2):
        op = random.choice(["subtree", "one_point", "size_fair", "uniform", "context_preserved"])
        if op == "subtree":
            return gp.cxOnePoint(ind1, ind2)
        elif op == "one_point":
            return gp.cxOnePointLeafBiased(ind1, ind2, termpb=0.1)
        elif op == "size_fair":
            return cxSizeFair(ind1, ind2)
        elif op == "uniform":
            return cxGPUniform(ind1, ind2)
        else:
            return cxContextPreserved(ind1, ind2)

    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

    # Custom mutation operators
    def mutHoist(individual):
        if len(individual) < 3:
            return individual,
        idx = random.randint(1, len(individual) - 1)
        slice_ = individual.searchSubtree(idx)
        return individual.__class__(individual[slice_]),
        
    def mutPermutation(individual):
        if len(individual) < 2:
            return individual,
        primitives = [i for i, node in enumerate(individual) if node.arity > 1]
        if not primitives:
            return individual,
        idx = random.choice(primitives)
        node = individual[idx]
        slices = []
        child_idx = idx + 1
        for _ in range(node.arity):
            s = individual.searchSubtree(child_idx)
            slices.append(s)
            child_idx = s.stop
        children = [individual[s] for s in slices]
        random.shuffle(children)
        new_tree = individual[:idx+1]
        for c in children:
            new_tree.extend(c)
        new_tree.extend(individual[slices[-1].stop:])
        return individual.__class__(new_tree),

    def mut_random(individual):
        op = random.choice(["subtree", "hoist", "node_replace", "permutation", "shrink"])
        if op == "subtree":
            return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)
        elif op == "hoist":
            return mutHoist(individual)
        elif op == "node_replace":
            return gp.mutNodeReplacement(individual, pset=pset)
        elif op == "permutation":
            return mutPermutation(individual)
        else:
            return gp.mutShrink(individual)

    toolbox.register("mate", mate_random)
    toolbox.register("mutate", mut_random)

    # Limit tree depth
    toolbox.decorate(
        "mate", gp.staticLimit(key=lambda x: x.height, max_value=max_depth)
    )
    toolbox.decorate(
        "mutate", gp.staticLimit(key=lambda x: x.height, max_value=max_depth)
    )

    all_mse = []
    best_expressions = []
    evals_per_run = []
    hit_evals = []

    for run in tqdm(range(n_runs), desc=f"[{equation_id}] GP Baseline", unit="run"):
        random.seed(seed + run)
        np.random.seed(seed + run)

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)

        try:
            run_evals = 0
            # Custom evolution loop to support early stopping
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            run_evals += len(invalid_ind)
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            hof.update(pop)
            if stats:
                stats.compile(pop)

            while (run_evals < max_evals
                   and hof[0].fitness.values[0] >= hit_threshold):
                # 1. Select k = 3 individuals at random
                indices = random.sample(range(len(pop)), 3)
                # Sort indices by the MSE (raw fitness value) in ascending order.
                # Smaller MSE is better. pop[indices[0]] is parent1, pop[indices[1]] is parent2, pop[indices[2]] is worst.
                indices.sort(key=lambda idx: pop[idx].fitness.values[0])
                parent1_idx, parent2_idx, worst_idx = indices
                parent1 = pop[parent1_idx]
                parent2 = pop[parent2_idx]

                # 2. Recombine the parents to produce one offspring
                # mate returns a tuple of two offspring; choose the first one
                off1, _ = toolbox.mate(toolbox.clone(parent1), toolbox.clone(parent2))
                offspring = off1

                # 3. Mutate the offspring with probability mut_prob (0.5)
                if random.random() < mut_prob:
                    offspring, = toolbox.mutate(offspring)

                # 4. Evaluate the offspring (exactly 1 evaluation)
                offspring.fitness.values = toolbox.evaluate(offspring)
                run_evals += 1

                # 5. Replace the worst of the 3 selected individuals in the population
                pop[worst_idx] = offspring

                # 6. Update Hall of Fame and stats
                hof.update([offspring])
                if stats:
                    stats.compile(pop)
        except Exception:
            all_mse.append(float("inf"))
            best_expressions.append("ERROR")
            evals_per_run.append(0)
            continue

        best_ind = hof[0]
        best_fit = best_ind.fitness.values[0]
        all_mse.append(best_fit)
        best_expressions.append(str(best_ind))
        evals_per_run.append(run_evals)
        if best_fit < hit_threshold:
            hit_evals.append(run_evals)

    n_hits = sum(1 for m in all_mse if m < hit_threshold)
    best_idx = int(np.argmin(all_mse))

    avg_evals = None
    if n_hits > 0:
        avg_evals = sum(evals_per_run) / n_hits

    return GPResult(
        equation_id=equation_id,
        best_mse=min(all_mse),
        best_expression=best_expressions[best_idx],
        median_mse=float(np.median(all_mse)),
        mean_mse=float(np.mean([m for m in all_mse if np.isfinite(m)])),
        n_hits=n_hits,
        avg_evals_to_hit=avg_evals,
        all_mse=all_mse,
    )
