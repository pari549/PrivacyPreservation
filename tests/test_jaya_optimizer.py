"""Checks Jaya convergence and bounded normalized priority weights."""
import unittest
import numpy as np
from src.optimization.jaya_optimizer import jaya_optimize, normalize_bounded_weights
from src.optimization.objective_functions import feature_diversity_score

class JayaOptimizerTest(unittest.TestCase):
    def test_fitness_improves_with_bounded_normalized_weights(self) -> None:
        initial = np.asarray([[.1, .1, .25, .25, .25, .25], [.2, .2, .25, .25, .25, .25],
                              [.3, .3, .25, .25, .25, .25], [.4, .4, .25, .25, .25, .25]])
        def repair(candidate: np.ndarray) -> np.ndarray:
            candidate[-4:] = normalize_bounded_weights(candidate[-4:])
            return candidate
        result = jaya_optimize(lambda x: float(x[0] + x[1]), 6, (0, 1), 4, 30, 7, initial, repair)
        self.assertGreater(result["best_fitness"], .2)
        self.assertTrue(np.all(result["best_solution"] >= 0))
        self.assertTrue(np.all(result["best_solution"] <= 1))
        self.assertEqual(len(result["convergence_history"]), 31)
        self.assertGreaterEqual(result["convergence_history"][-1], result["convergence_history"][0])
        weights = result["best_solution"][-4:]
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        self.assertTrue(np.all(weights >= .10))
        self.assertTrue(np.all(weights <= .70))
        self.assertGreaterEqual(feature_diversity_score(weights), 0.0)

if __name__ == "__main__": unittest.main()
