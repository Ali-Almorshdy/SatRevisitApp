# gaoptm.py
"""
GA optimizer for satellite revisit minimization.


run_ga(...) returns (best_coes, best_score, history)
 - best_coes: np.ndarray shape (num_sats,6) in UI/GA order [a,e,TA,RA,incl,w]
 - best_score: float (mean revisit time in seconds)
 - history: list[float] per-generation best
"""

import numpy as np
from datetime import datetime
from typing import Callable, List, Tuple, Optional
from passes import compute_passes



# Default large penalty for invalid individuals
PENALTY = 1e9

def _flatten_key(arr: np.ndarray, epoch: Optional[datetime], dt: float, duration_days: float):
    f = np.round(np.asarray(arr).reshape(-1), 6)
    e_ts = None if epoch is None else int(epoch.timestamp())
    return (tuple(f.tolist()), e_ts, float(dt), float(duration_days))

def run_ga(
    locations: np.ndarray,
    num_sats: int = 3,
    pop_size: int = 20,
    generations: int = 30,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.7,
    duration_days: float = 1.0,
    dt: float = 30.0,
    seed: Optional[int] = None,
    sma_bounds: Tuple[float,float] = (6800.0, 7500.0),
    e_bounds: Tuple[float,float] = (0.0, 0.01),
    incl_bounds: Tuple[float,float] = (0.0, 98.0),
    ra_bounds: Tuple[float,float] = (0.0, 360.0),
    ta_bounds: Tuple[float,float] = (0.0, 360.0),
    w_bounds: Tuple[float,float] = (0.0, 360.0),
    epoch: Optional[datetime] = None,
    gap_seconds: float = 20.0,
    stop_flag: Callable[[], bool] = lambda: False,
    progress_callback: Optional[Callable[[int, float, np.ndarray, List[float]], None]] = None,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Run GA.

    Parameters:
      - locations: ndarray (M,2) lat,lon in degrees
      - num_sats: int
      - pop_size: int
      - generations: int
      - mutation_rate: prob per individual to mutate (will mutate a single gene in that case)
      - crossover_rate: prob to crossover parents
      - duration_days, dt: propagation control for compute_passes during GA (coarser dt speeds things)
      - sma_bounds, e_bounds, incl_bounds, ra_bounds, ta_bounds, w_bounds: tuple (min,max) bounds in UI/GA order
      - epoch: datetime or None
      - stop_flag: callable returning True to stop early
      - progress_callback: optional callable called after each generation: (gen, best_score, best_coes, history)
    Returns:
      (best_coes (np.ndarray num_satsx6), best_score(float), history(list))
    """

    if seed is not None:
        np.random.seed(int(seed))

    locations = np.asarray(locations, dtype=float)

    # bounds in UI/GA element order: [a, e, TA, RA, incl, w]
    def random_one():
        return np.array([
            np.random.uniform(*sma_bounds),
            np.random.uniform(*e_bounds),
            np.random.uniform(*ta_bounds),
            np.random.uniform(*ra_bounds),
            np.random.uniform(*incl_bounds),
            np.random.uniform(*w_bounds)
        ], dtype=float)

    def random_individual():
        return np.vstack([random_one() for _ in range(num_sats)])  # shape (num_sats,6)

    # caching to avoid recompute identical individuals
    eval_cache = {}

    def evaluate(individual: np.ndarray) -> float:
        """
        Evaluate an individual (num_sats,6) in UI/GA order [a,e,TA,RA,incl,w].
        Returns mean revisit time in seconds (lower is better) or large PENALTY on failure.
        """
        key = _flatten_key(individual, epoch, dt, duration_days)
        if key in eval_cache:
            return eval_cache[key]

        try:
            # compute_passes is expected to accept coes in same UI/GA order
            res = compute_passes(locations, individual, epoch=epoch if epoch is not None else datetime.utcnow(),
                                 dt=float(dt), duration_days=float(duration_days), gap_seconds=float(gap_seconds))
            mean_rev = res.get("global_stats", {}).get("meanRev", None)
            if mean_rev is None or np.isnan(mean_rev):
                score = PENALTY
            else:
                score = float(mean_rev)
        except Exception:
            score = PENALTY

        eval_cache[key] = score
        return score

    # initialize population
    population: List[np.ndarray] = [random_individual() for _ in range(pop_size)]
    scores: List[float] = [evaluate(ind) for ind in population]

    best_idx = int(np.argmin(scores))
    best_ind = population[best_idx].copy()
    best_score = float(scores[best_idx])
    history: List[float] = [best_score]

    # progress callback initial
    if progress_callback is not None:
        try:
            progress_callback(0, best_score, best_ind.copy(), history.copy())
        except Exception:
            pass

    # GA loop
    for gen in range(1, generations + 1):
        if stop_flag():
            # Stop requested — return current best
            break

        # Selection: tournament (size 2)
        parents = []
        for _ in range(pop_size):
            i, j = np.random.randint(0, pop_size, 2)
            parents.append(population[i] if scores[i] < scores[j] else population[j])

        # Crossover -> children generation (generational replacement)
        children: List[np.ndarray] = []
        for i in range(0, pop_size, 2):
            p1 = parents[i]
            p2 = parents[i+1] if (i+1) < len(parents) else parents[0]
            flat1 = p1.flatten()
            flat2 = p2.flatten()
            if np.random.rand() < crossover_rate:
                pt = np.random.randint(1, flat1.size)
                c1_flat = np.hstack([flat1[:pt], flat2[pt:]])
                c2_flat = np.hstack([flat2[:pt], flat1[pt:]])
            else:
                c1_flat = flat1.copy()
                c2_flat = flat2.copy()
            children.append(c1_flat.reshape((num_sats, 6)))
            children.append(c2_flat.reshape((num_sats, 6)))

        
        gene_mut_rate = mutation_rate  # direct mapping
        for child in children:
            mask = (np.random.rand(child.shape[0], child.shape[1]) < gene_mut_rate)
            if not np.any(mask):
                continue
            # mutate positions according to mask
            for s_idx in range(child.shape[0]):
                for p_idx in range(child.shape[1]):
                    if not mask[s_idx, p_idx]:
                        continue
                    if p_idx == 0:
                        child[s_idx, p_idx] = np.random.uniform(*sma_bounds)
                    elif p_idx == 1:
                        child[s_idx, p_idx] = np.random.uniform(*e_bounds)
                    elif p_idx == 2:
                        child[s_idx, p_idx] = np.random.uniform(*ta_bounds)
                    elif p_idx == 3:
                        child[s_idx, p_idx] = np.random.uniform(*ra_bounds)
                    elif p_idx == 4:
                        child[s_idx, p_idx] = np.random.uniform(*incl_bounds)
                    elif p_idx == 5:
                        child[s_idx, p_idx] = np.random.uniform(*w_bounds)

        # Evaluate children
        new_scores = [evaluate(ch) for ch in children]

        # Replace population (generational)
        population = children
        scores = new_scores

        # Keep best seen so far (elitism)
        gen_best_idx = int(np.argmin(scores))
        gen_best = population[gen_best_idx].copy()
        gen_score = float(scores[gen_best_idx])

        if gen_score < best_score:
            best_score = gen_score
            best_ind = gen_best.copy()

        history.append(best_score)

        # call progress callback if provided
        if progress_callback is not None:
            try:
                progress_callback(gen, best_score, best_ind.copy(), history.copy())
            except Exception:
                pass

    # Ensure returned best_coes is np.ndarray shape (num_sats,6)
    best_coes = np.asarray(best_ind, dtype=float).reshape((num_sats, 6))
    return best_coes, float(best_score), history
