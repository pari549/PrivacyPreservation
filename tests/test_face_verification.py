"""Synthetic embedding and failure tests; these are not face-recognition results."""
import unittest
from dataclasses import replace

import numpy as np

from src.evaluation.face_verification import (
    ImageProcessingRecord,
    bootstrap_roc_auc_by_identity,
    calibrate_verification_threshold,
    compare_verification_metrics,
    evaluate_verification,
    normalize_embeddings,
    score_manifest_pairs,
)
from src.evaluation.identity_pairs import (
    generate_pair_manifest,
    split_calibration_scoring_identities,
    split_identities,
)


def context(pair_count=2):
    inventory = {f"person-{i:02d}": [f"p{i:02d}-img{j}.png" for j in range(4)] for i in range(16)}
    split = split_identities(inventory, dataset_identifier="synthetic", dataset_release="test", seed=10,
                             counts={"development": 4, "optimization": 4, "validation": 4, "final_test": 4})
    subsets = split_calibration_scoring_identities(split, "validation", seed=24, calibration_count=2)
    calibration = generate_pair_manifest(inventory, split, partition="validation", subset_role="calibration",
                                         pair_count_per_class=pair_count, seed=101,
                                         scenario="scenario_a_both_transformed", subsets=subsets)
    scoring = generate_pair_manifest(inventory, split, partition="validation", subset_role="scoring",
                                     pair_count_per_class=pair_count, seed=102,
                                     scenario="scenario_a_both_transformed", subsets=subsets)
    return inventory, split, subsets, calibration, scoring


def identity_embeddings(inventory):
    identities = sorted(inventory)
    basis = np.eye(len(identities), dtype=np.float64)
    return {image: basis[index].copy() for index, identity in enumerate(identities) for image in inventory[identity]}


class FaceVerificationTest(unittest.TestCase):
    def test_identity_dyadic_bootstrap_is_deterministic_and_reports_interval(self):
        inventory, _, _, _, manifest = context(pair_count=2)
        scoring = score_manifest_pairs(manifest, identity_embeddings(inventory))
        first = bootstrap_roc_auc_by_identity(scoring, bootstrap_replicates=300, seed=117,
                                              minimum_valid_fraction=0.3)
        second = bootstrap_roc_auc_by_identity(scoring, bootstrap_replicates=300, seed=117,
                                               minimum_valid_fraction=0.3)
        self.assertEqual(first, second)
        self.assertEqual(first.method, "identity_dyadic_vertex_bootstrap_percentile_v1")
        self.assertEqual(first.point_estimate, 1.0)
        self.assertEqual(first.bootstrap_replicates, 300)
        self.assertEqual(first.valid_replicates + first.invalid_replicates, 300)
        self.assertEqual(first.status, "valid")
        self.assertIsNotNone(first.lower_bound)
        self.assertLessEqual(first.lower_bound, first.upper_bound)
        self.assertEqual(first.confidence_level, 0.95)

    def test_identity_bootstrap_exposes_invalid_replicates_and_withholds_interval(self):
        inventory, _, _, _, manifest = context(pair_count=1)  # two scoring identities
        scoring = score_manifest_pairs(manifest, identity_embeddings(inventory))
        result = bootstrap_roc_auc_by_identity(scoring, bootstrap_replicates=100, seed=9)
        self.assertEqual(result.status, "insufficient_valid_replicates")
        self.assertGreater(result.invalid_replicates, 0)
        self.assertIsNone(result.lower_bound)
        self.assertIsNone(result.upper_bound)
        self.assertIn("below the required minimum", result.limitations[0])

    def test_identity_bootstrap_handles_class_imbalance_and_invalid_point_estimate(self):
        inventory, _, _, _, manifest = context(pair_count=2)
        scoring = score_manifest_pairs(manifest, identity_embeddings(inventory))
        impostor_id = next(pair.pair_id for pair in manifest.pairs if pair.label == "impostor")
        scores = tuple(replace(score, status="unscoreable", similarity=None,
                               reason="synthetic missing endpoint") if score.pair_id == impostor_id else score
                       for score in scoring.scores)
        imbalanced = replace(scoring, scores=scores)
        self.assertEqual(sum(score.status == "scoreable" and score.label == "genuine"
                             for score in imbalanced.scores), 2)
        self.assertEqual(sum(score.status == "scoreable" and score.label == "impostor"
                             for score in imbalanced.scores), 1)
        interval = bootstrap_roc_auc_by_identity(imbalanced, bootstrap_replicates=300, seed=117,
                                                 minimum_valid_fraction=0.3)
        self.assertEqual(interval.point_estimate, 1.0)
        self.assertIn(interval.status, ("valid", "insufficient_valid_replicates"))
        self.assertEqual(interval.valid_replicates + interval.invalid_replicates, 300)

        no_impostor_scores = tuple(replace(score, status="unscoreable", similarity=None)
                                   if score.label == "impostor" else score for score in scoring.scores)
        undefined = bootstrap_roc_auc_by_identity(replace(scoring, scores=no_impostor_scores),
                                                  bootstrap_replicates=20, seed=2)
        self.assertEqual(undefined.status, "invalid_point_estimate")
        self.assertIsNone(undefined.point_estimate)
        self.assertEqual(undefined.invalid_replicates, 20)

    def test_identity_bootstrap_validates_parameters_and_pair_identity_binding(self):
        inventory, _, _, _, manifest = context(pair_count=1)
        scoring = score_manifest_pairs(manifest, identity_embeddings(inventory))
        for kwargs in ({"confidence_level": 1.0}, {"bootstrap_replicates": 0},
                       {"seed": -1}, {"minimum_valid_fraction": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bootstrap_roc_auc_by_identity(scoring, **kwargs)
        bad_label = replace(scoring, scores=(replace(scoring.scores[0], label="impostor"),) + scoring.scores[1:])
        with self.assertRaises(ValueError):
            bootstrap_roc_auc_by_identity(bad_label, bootstrap_replicates=10)

    def test_embedding_normalization_and_cosine_scores(self):
        normalized, statuses = normalize_embeddings({"a": [3.0, 4.0], "b": [0.0, 2.0]})
        np.testing.assert_allclose(normalized["a"], [0.6, 0.8])
        np.testing.assert_allclose(normalized["b"], [0.0, 1.0])
        self.assertTrue(all(item.status == "valid" for item in statuses))
        inventory, split, _, _, scoring = context(pair_count=1)
        result = score_manifest_pairs(scoring, identity_embeddings(inventory))
        for item in result.scores:
            expected = 1.0 if item.label == "genuine" else 0.0
            self.assertAlmostEqual(item.similarity, expected)

    def test_invalid_zero_nonfinite_missing_and_inconsistent_embeddings_reported(self):
        _, statuses = normalize_embeddings({"zero": [0.0, 0.0], "nan": [1.0, np.nan],
                                            "short": [1.0], "long": [1.0, 2.0]})
        by_id = {item.image_id: item for item in statuses}
        self.assertEqual(by_id["zero"].status, "invalid")
        self.assertIn("zero", by_id["zero"].reason)
        self.assertEqual(by_id["nan"].status, "invalid")
        self.assertIn("inconsistent", by_id["short"].reason)
        self.assertIn("inconsistent", by_id["long"].reason)
        inventory, split, _, _, scoring = context(pair_count=1)
        embeds = identity_embeddings(inventory)
        referenced = scoring.pairs[0].first_image_id
        del embeds[referenced]
        result = score_manifest_pairs(scoring, embeds)
        self.assertTrue(any(item.image_id == referenced and item.status == "missing" for item in result.embedding_statuses))
        self.assertGreater(result.failure_accounting.missing_pair_endpoints, 0)
        self.assertTrue(any(pair.status == "unscoreable" for pair in result.scores))

    def test_roc_auc_and_calibrated_tar_far(self):
        inventory, split, subsets, calibration_manifest, scoring_manifest = context(pair_count=2)
        embeddings = identity_embeddings(inventory)
        calibration_scores = score_manifest_pairs(calibration_manifest, embeddings)
        calibration = calibrate_verification_threshold(calibration_scores, target_far=0.0,
                                                        minimum_pair_coverage=1.0)
        self.assertEqual(calibration.status, "valid")
        self.assertEqual(calibration.achieved_far, 0.0)
        self.assertEqual(calibration.achieved_tar, 1.0)
        scoring = score_manifest_pairs(scoring_manifest, embeddings)
        metrics = evaluate_verification(scoring, calibration, minimum_pair_coverage=1.0)
        self.assertEqual(metrics.roc_auc, 1.0)
        self.assertEqual(metrics.metric_direction, "higher_is_better")
        self.assertEqual(metrics.achieved_far, 0.0)
        self.assertEqual(metrics.achieved_tar, 1.0)
        self.assertTrue(metrics.suitable_for_privacy_claim)

    def test_tied_calibration_scores_have_deterministic_threshold(self):
        inventory, split, _, calibration_manifest, _ = context(pair_count=2)
        embeddings = {image: np.ones(3) for images in inventory.values() for image in images}
        scored = score_manifest_pairs(calibration_manifest, embeddings)
        first = calibrate_verification_threshold(scored, target_far=1.0, minimum_pair_coverage=1.0)
        second = calibrate_verification_threshold(scored, target_far=1.0, minimum_pair_coverage=1.0)
        self.assertEqual(first.threshold, second.threshold)
        self.assertEqual(first.achieved_far, 1.0)
        self.assertEqual(first.achieved_tar, 1.0)

    def test_calibration_scoring_separation_is_enforced(self):
        inventory, split, _, calibration_manifest, scoring_manifest = context(pair_count=1)
        embeddings = identity_embeddings(inventory)
        scoring_from_cal_manifest = score_manifest_pairs(scoring_manifest, embeddings)
        with self.assertRaises(ValueError):
            calibrate_verification_threshold(scoring_from_cal_manifest, target_far=0.0, minimum_pair_coverage=1.0)
        calibration = calibrate_verification_threshold(score_manifest_pairs(calibration_manifest, embeddings),
                                                       target_far=0.0, minimum_pair_coverage=1.0)
        metrics = evaluate_verification(score_manifest_pairs(scoring_manifest, embeddings), calibration,
                                        minimum_pair_coverage=1.0)
        self.assertEqual(metrics.subset_role, "scoring")

    def test_coverage_failure_is_explicit_and_not_privacy_success(self):
        inventory, split, _, calibration_manifest, scoring_manifest = context(pair_count=2)
        embeddings = identity_embeddings(inventory)
        calibration = calibrate_verification_threshold(score_manifest_pairs(calibration_manifest, embeddings),
                                                       target_far=0.0, minimum_pair_coverage=1.0)
        missing_id = scoring_manifest.pairs[0].first_image_id
        del embeddings[missing_id]
        outcomes = {
            missing_id: ImageProcessingRecord(missing_id, True, detection_status="no_face",
                                              alignment_status="not_attempted", embedding_status="missing",
                                              inference_status="not_attempted")
        }
        result = score_manifest_pairs(scoring_manifest, embeddings, image_outcomes=outcomes)
        metrics = evaluate_verification(result, calibration, minimum_pair_coverage=1.0)
        self.assertFalse(metrics.suitable_for_privacy_claim)
        self.assertGreater(result.failure_accounting.no_face_detections, 0)
        self.assertGreater(result.failure_accounting.unscoreable_genuine_pairs
                           + result.failure_accounting.unscoreable_impostor_pairs, 0)
        self.assertTrue(any("coverage" in item for item in metrics.limitations))

    def test_failure_accounting_categories_are_separate(self):
        inventory, split, _, _, scoring_manifest = context(pair_count=1)
        embeds = identity_embeddings(inventory)
        refs = {pair.first_image_id for pair in scoring_manifest.pairs} | {pair.second_image_id for pair in scoring_manifest.pairs}
        statuses = ["success", "no_face", "multiple_faces", "alignment_failure", "invalid_embedding", "inference_error"]
        records = {}
        outcome_ids = [f"extra-outcome-{index}" for index in range(len(statuses))] + sorted(refs)
        for image_id, status in zip(outcome_ids, statuses + ["unknown"] * len(refs)):
            records[image_id] = ImageProcessingRecord(
                image_id, True,
                detection_status={"success": "success", "no_face": "no_face", "multiple_faces": "multiple_faces",
                                  "alignment_failure": "success", "invalid_embedding": "success", "inference_error": "success"}.get(status, "success"),
                alignment_status="failure" if status == "alignment_failure" else "success",
                embedding_status="invalid" if status == "invalid_embedding" else "success",
                inference_status="error" if status == "inference_error" else "success")
        scored = score_manifest_pairs(scoring_manifest, embeds, image_outcomes=records)
        failure = scored.failure_accounting
        self.assertEqual(failure.images_attempted, len(outcome_ids))
        self.assertEqual(failure.no_face_detections, 1)
        self.assertEqual(failure.multiple_face_ambiguities, 1)
        self.assertEqual(failure.alignment_failures, 1)
        self.assertEqual(failure.invalid_embeddings, 1)
        self.assertEqual(failure.inference_errors, 1)
        self.assertTrue(failure.image_outcomes_complete)

    def test_comparison_uses_chance_adjusted_advantage_and_flags_invalid_baseline(self):
        inventory, split, _, calibration_manifest, scoring_manifest = context(pair_count=2)
        original_embeddings = identity_embeddings(inventory)
        calibration = calibrate_verification_threshold(score_manifest_pairs(calibration_manifest, original_embeddings),
                                                       target_far=0.0, minimum_pair_coverage=1.0)
        original = evaluate_verification(score_manifest_pairs(scoring_manifest, original_embeddings), calibration,
                                         minimum_pair_coverage=1.0)
        transformed_embeddings = {image: np.ones(4) for images in inventory.values() for image in images}
        transformed_calibration = calibrate_verification_threshold(
            score_manifest_pairs(calibration_manifest, transformed_embeddings, condition_identifier="transformed"),
            target_far=0.0, minimum_pair_coverage=1.0)
        transformed = evaluate_verification(
            score_manifest_pairs(scoring_manifest, transformed_embeddings, condition_identifier="transformed"),
            transformed_calibration,
                                            minimum_pair_coverage=1.0)
        comparison = compare_verification_metrics(original, transformed)
        self.assertEqual(comparison.original_roc_auc, 1.0)
        self.assertEqual(comparison.transformed_roc_auc, 0.5)
        self.assertEqual(comparison.chance_adjusted_attack_advantage_reduction, 1.0)
        self.assertTrue(comparison.suitable_for_privacy_claim)
        chance_baseline = replace_metric_auc(original, 0.5)
        invalid = compare_verification_metrics(chance_baseline, transformed)
        self.assertIsNone(invalid.chance_adjusted_attack_advantage_reduction)
        self.assertFalse(invalid.suitable_for_privacy_claim)

    def test_threshold_cannot_cross_evaluation_conditions(self):
        inventory, split, _, calibration_manifest, scoring_manifest = context(pair_count=1)
        embeddings = identity_embeddings(inventory)
        calibration = calibrate_verification_threshold(
            score_manifest_pairs(calibration_manifest, embeddings, condition_identifier="original"),
            target_far=0.0, minimum_pair_coverage=1.0)
        scoring = score_manifest_pairs(scoring_manifest, embeddings, condition_identifier="transformed")
        with self.assertRaises(ValueError):
            evaluate_verification(scoring, calibration, minimum_pair_coverage=1.0)


def replace_metric_auc(metric, auc):
    from dataclasses import replace
    return replace(metric, roc_auc=auc)


if __name__ == "__main__":
    unittest.main()
