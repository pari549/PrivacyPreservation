"""Run and compare legacy and balanced Jaya objective functions."""
from __future__ import annotations
import json
import os
from pathlib import Path

from src.preprocessing.data_loader import DEFAULT_LOG_PATH, PROJECT_ROOT, get_dataset_logger

MATPLOTLIB_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.optimization.jaya_optimizer import jaya_optimize, normalize_bounded_weights
from src.optimization.objective_functions import block_priorities, feature_diversity_score, balanced_multiobjective, legacy_method1_objective, method2_privacy_objective

INPUT_PATH = PROJECT_ROOT / "results" / "metrics" / "block_feature_matrix.json"
RESULT_PATH = PROJECT_ROOT / "results" / "metrics" / "jaya_results.json"
IMAGE_DIR = PROJECT_ROOT / "results" / "images" / "jaya"
WEIGHT_NAMES = ("variance", "gradient", "entropy", "roi")

def run_jaya(input_path: Path = INPUT_PATH, result_path: Path = RESULT_PATH,
             image_dir: Path = IMAGE_DIR, log_path: Path = DEFAULT_LOG_PATH,
             population_size: int = 20, iterations: int = 100, seed: int = 42) -> dict:
    logger = get_dataset_logger(log_path)
    logger.info("Running Jaya legacy comparison and balanced multi-objective search.")
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    samples = payload["samples"]
    features = [block for sample in samples for block in sample["features"]]
    n = len(features)
    if not n: raise ValueError("Feature matrix contains no blocks")
    dimensions = 2*n + 4
    lower = np.r_[np.zeros(2*n), np.full(4, .10)]
    upper = np.r_[np.ones(2*n), np.full(4, .70)]
    rng = np.random.default_rng(seed)
    initial = rng.uniform(0, 1, size=(population_size, dimensions))
    initial[0, :2*n] = .5; initial[0, -4:] = .25
    if population_size > 1:
        initial[1:, -4:] = rng.dirichlet(np.ones(4), size=population_size-1)
    def repair(vector: np.ndarray) -> np.ndarray:
        fixed = vector.copy(); fixed[-4:] = normalize_bounded_weights(fixed[-4:]); return fixed
    old_obj = lambda x: legacy_method1_objective(x, n, features)
    balanced_obj = lambda x: balanced_multiobjective(x, n, features)
    privacy_obj = lambda x: method2_privacy_objective(x, n, features)
    old = jaya_optimize(old_obj, dimensions, (lower, upper), population_size, iterations, seed, initial, repair)
    balanced = jaya_optimize(balanced_obj, dimensions, (lower, upper), population_size, iterations, seed, initial, repair)
    privacy = jaya_optimize(privacy_obj, dimensions, (lower, upper), population_size, iterations, seed+1, initial, repair)

    def pack(outcome: dict) -> dict:
        solution = outcome["best_solution"]
        weights = normalize_bounded_weights(solution[-4:])
        return {"optimized_alpha": solution[:n].tolist(), "optimized_beta": solution[n:2*n].tolist(),
                "optimized_weights": {key: float(value) for key, value in zip(WEIGHT_NAMES, weights)},
                "feature_diversity_score": feature_diversity_score(weights),
                "best_fitness": outcome["best_fitness"], "iterations": iterations,
                "convergence_history": outcome["convergence_history"],
                "weight_history": outcome["best_solution_history"][:, -4:].tolist()}
    old_result, new_result, method2 = pack(old), pack(balanced), pack(privacy)
    result = {
        "objective_note": "Recovery, utility, and privacy scores are illustrative optimization proxies, not measured outcomes; no image transform is applied.",
        "balanced_fitness": "mean(recovery_proxy, utility_proxy, privacy_proxy, feature_diversity_score)",
        "feature_diversity_formula": "1 - sum(w_i^2)",
        "decision_variables": {"alpha_bounds": [0.0, 1.0], "beta_bounds": [0.0, 1.0],
            "weight_bounds": [0.10, 0.70], "weight_sum": 1.0, "block_count": n},
        "optimized_alpha": new_result["optimized_alpha"], "optimized_beta": new_result["optimized_beta"],
        "optimized_weights": new_result["optimized_weights"],
        "feature_diversity_score": new_result["feature_diversity_score"],
        "best_fitness": new_result["best_fitness"], "iterations": iterations,
        "balanced_objective": new_result,
        "legacy_objective": old_result,
        "method2_privacy_only": method2,
        "convergence_comparison": {"legacy": old_result["convergence_history"], "balanced": new_result["convergence_history"]},
    }
    result_path = Path(result_path); result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    image_dir = Path(image_dir); image_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(old_result["convergence_history"], label="Previous objective")
    ax.plot(new_result["convergence_history"], label="Balanced multi-objective")
    ax.set(title="Jaya fitness comparison", xlabel="Iteration", ylabel="Best fitness")
    ax.legend(); fig.tight_layout(); fig.savefig(image_dir / "fitness_comparison.png", dpi=180); plt.close(fig)

    weight_history = np.asarray(new_result["weight_history"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, name in enumerate(WEIGHT_NAMES): ax.plot(weight_history[:, index], label=name)
    ax.axhline(.10, color="gray", linestyle="--", linewidth=.8); ax.axhline(.70, color="gray", linestyle="--", linewidth=.8)
    ax.set(title="Priority weight evolution", xlabel="Iteration", ylabel="Normalized weight", ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(image_dir / "weight_evolution.png", dpi=180); plt.close(fig)

    alpha_grid = np.asarray(new_result["optimized_alpha"]).reshape(len(samples), -1)
    priority_grid = block_priorities(features, np.asarray(list(new_result["optimized_weights"].values()))).reshape(len(samples), -1)
    for filename, grid, title in (("alpha_heatmap.png", alpha_grid, "Optimized alpha"), ("priority_heatmap.png", priority_grid, "Priority score")):
        fig, ax = plt.subplots(figsize=(10, 4)); plot = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0, vmax=1 if filename.startswith("alpha") else None)
        ax.set(title=title, xlabel="Block index within sample", ylabel="Representative sample")
        fig.colorbar(plot, ax=ax, label=title); fig.tight_layout(); fig.savefig(image_dir / filename, dpi=180); plt.close(fig)
    logger.info("Balanced Jaya results saved to %s", result_path)
    return result

if __name__ == "__main__": run_jaya()
