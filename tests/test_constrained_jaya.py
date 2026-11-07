import unittest

import numpy as np

from src.optimization.jaya_optimizer import (
    CandidateEvaluation,
    UtilityConstraint,
    jaya_optimize_constrained,
    rank_candidate,
)


class ConstrainedJayaTest(unittest.TestCase):
    def setUp(self):
        self.constraints = (
            UtilityConstraint("ssim", "minimum", 0.8, 0.1),
            UtilityConstraint("mae", "maximum", 0.2, 0.1),
        )

    @staticmethod
    def evaluation(score, ssim, mae):
        return CandidateEvaluation(score, {"ssim": ssim, "mae": mae})

    def test_feasibility_precedes_fitness_and_violation_is_scaled(self):
        feasible = rank_candidate(self.evaluation(-10, 0.8, 0.2), self.constraints)
        infeasible_high_score = rank_candidate(self.evaluation(100, 0.7, 0.2), self.constraints)
        self.assertTrue(feasible.feasible)
        self.assertEqual(feasible.total_normalized_violation, 0.0)
        self.assertGreater(feasible.sort_key, infeasible_high_score.sort_key)
        self.assertAlmostEqual(infeasible_high_score.total_normalized_violation, 1.0)

    def test_less_infeasible_candidate_beats_more_infeasible_candidate(self):
        lower = rank_candidate(self.evaluation(1, 0.79, 0.2), self.constraints)
        higher = rank_candidate(self.evaluation(100, 0.6, 0.2), self.constraints)
        self.assertGreater(lower.sort_key, higher.sort_key)

    def test_feasible_tie_break_uses_higher_privacy_fitness(self):
        low = rank_candidate(self.evaluation(0.2, 0.9, 0.1), self.constraints)
        high = rank_candidate(self.evaluation(0.3, 0.9, 0.1), self.constraints)
        self.assertGreater(high.sort_key, low.sort_key)

    def test_invalid_candidate_ranks_below_valid_candidate(self):
        invalid = rank_candidate(CandidateEvaluation.invalid("attack unavailable"), self.constraints)
        valid = rank_candidate(self.evaluation(-100, 0.9, 0.1), self.constraints)
        self.assertGreater(valid.sort_key, invalid.sort_key)

    def test_malformed_and_nonfinite_records_are_rejected(self):
        for record in (
            CandidateEvaluation(float("nan"), {"ssim": 1, "mae": 0}),
            CandidateEvaluation(1, {"ssim": float("inf"), "mae": 0}),
            CandidateEvaluation(1, {"ssim": 1}),
            CandidateEvaluation.invalid("bad"),
        ):
            if not record.valid:
                continue
            with self.subTest(record=record), self.assertRaises(ValueError):
                rank_candidate(record, self.constraints)
        with self.assertRaises(TypeError):
            rank_candidate(object(), self.constraints)

    def test_optimizer_is_reproducible_and_returns_feasible_candidate(self):
        def evaluator(theta):
            x = float(theta[0])
            return self.evaluation(-(x - 0.7) ** 2, 0.85 + 0.1 * x, 0.1)

        args = dict(evaluator=evaluator, dimensions=1, constraints=self.constraints,
                    population_size=8, iterations=5, seed=123)
        first = jaya_optimize_constrained(**args)
        second = jaya_optimize_constrained(**args)
        np.testing.assert_array_equal(first["best_solution"], second["best_solution"])
        self.assertEqual(first["best_fitness"], second["best_fitness"])
        self.assertTrue(first["best_rank"]["feasible"])
        self.assertEqual(len(first["convergence_history"]), 6)

    def test_evaluator_failures_and_nonfinite_results_get_no_fitness(self):
        def failed(_):
            raise RuntimeError("evaluation failed")

        result = jaya_optimize_constrained(
            failed, 1, self.constraints, population_size=2, iterations=1, seed=1
        )
        self.assertIsNone(result["best_fitness"])
        self.assertEqual(result["best_rank"]["tier"], 0)
        self.assertIn("RuntimeError", result["best_rank"]["invalid_reason"])

        def nonfinite(_):
            return self.evaluation(float("nan"), 1, 0)

        result = jaya_optimize_constrained(
            nonfinite, 1, self.constraints, population_size=2, iterations=1, seed=1
        )
        self.assertIsNone(result["best_fitness"])

    def test_invalid_optimizer_configuration_is_rejected(self):
        evaluator = lambda _: self.evaluation(1, 1, 0)
        cases = (
            {"dimensions": 0},
            {"population_size": 1},
            {"iterations": 0},
            {"seed": -1},
            {"bounds": (0, 1, 2)},
            {"bounds": (1, 0)},
        )
        for override in cases:
            args = dict(evaluator=evaluator, dimensions=1, constraints=self.constraints,
                        population_size=2, iterations=1, seed=1)
            args.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                jaya_optimize_constrained(**args)


if __name__ == "__main__":
    unittest.main()
