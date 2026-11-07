"""Synthetic-only tests for identity-disjoint splitting and pair manifests."""
from dataclasses import replace
import json
import unittest

from src.evaluation.identity_pairs import (
    EvaluationIdentitySubsets,
    PairManifest,
    generate_pair_manifest,
    split_calibration_scoring_identities,
    split_identities,
    validate_identity_inventory,
)


def synthetic_inventory(count=16, images_per_identity=4):
    return {f"subject-{i:02d}": [f"image-{i:02d}-{j:02d}.png" for j in range(images_per_identity)]
            for i in range(count)}


def make_split(inventory=None):
    inventory = inventory or synthetic_inventory()
    return split_identities(
        inventory, dataset_identifier="synthetic-faces", dataset_release="test-v1", seed=729,
        counts={"development": 4, "optimization": 4, "validation": 4, "final_test": 4},
    )


class IdentityPairProtocolTest(unittest.TestCase):
    def test_identity_split_is_deterministic_and_disjoint(self):
        inventory = synthetic_inventory()
        first = make_split(inventory)
        second = make_split(dict(reversed(list(inventory.items()))))
        self.assertEqual(first.to_json(), second.to_json())
        all_parts = [identity for values in first.partitions.values() for identity in values]
        self.assertEqual(len(all_parts), len(set(all_parts)))
        self.assertEqual(set(all_parts), set(inventory))
        first.validate_inventory(inventory)
        self.assertEqual(type(first).from_json(first.to_json()).to_json(), first.to_json())

    def test_proportion_allocation_and_explicit_counts(self):
        inventory = synthetic_inventory(11)
        split = split_identities(inventory, dataset_identifier="synthetic", dataset_release="r1", seed=4,
                                 proportions={"development": .4, "optimization": .2,
                                              "validation": .2, "final_test": .2})
        self.assertEqual(sum(map(len, split.partitions.values())), 11)
        self.assertEqual([len(split.partitions[k]) for k in
                          ("development", "optimization", "validation", "final_test")], [5, 2, 2, 2])
        with self.assertRaises(ValueError):
            split_identities(inventory, dataset_identifier="synthetic", dataset_release="r1", seed=4,
                             counts={"development": 3, "optimization": 3, "validation": 3, "final_test": 3})

    def test_malformed_mapping_duplicate_images_and_identifiers_rejected(self):
        with self.assertRaises(ValueError):
            validate_identity_inventory({"id": ["same"], "id2": ["same"]})
        with self.assertRaises(ValueError):
            validate_identity_inventory({" id ": ["image"]})
        with self.assertRaises(ValueError):
            validate_identity_inventory({"id": []})

    def test_calibration_and_scoring_identities_are_disjoint(self):
        split = make_split()
        subsets = split_calibration_scoring_identities(split, "validation", seed=11, calibration_count=2)
        self.assertEqual(len(subsets.calibration_identities), 2)
        self.assertEqual(len(subsets.scoring_identities), 2)
        self.assertFalse(set(subsets.calibration_identities) & set(subsets.scoring_identities))
        self.assertEqual(set(subsets.calibration_identities) | set(subsets.scoring_identities),
                         set(split.partitions["validation"]))

    def test_pair_classes_are_correct_balanced_and_round_trip(self):
        inventory = synthetic_inventory()
        split = make_split(inventory)
        manifest = generate_pair_manifest(inventory, split, partition="final_test", subset_role="partition",
                                           pair_count_per_class=4, seed=21,
                                           scenario="scenario_a_both_transformed")
        self.assertEqual([p.label for p in manifest.pairs].count("genuine"), 4)
        self.assertEqual([p.label for p in manifest.pairs].count("impostor"), 4)
        for pair in manifest.pairs:
            self.assertEqual(pair.label == "genuine", pair.first_identity_id == pair.second_identity_id)
        restored = PairManifest.from_json(manifest.to_json())
        restored.validate(inventory, split)
        self.assertEqual(restored.to_json(), manifest.to_json())

    def test_scenario_b_roles_and_subset_membership(self):
        inventory = synthetic_inventory()
        split = make_split(inventory)
        subsets = split_calibration_scoring_identities(split, "validation", seed=3, calibration_count=2)
        manifest = generate_pair_manifest(inventory, split, partition="validation", subset_role="scoring",
                                          pair_count_per_class=2, seed=90,
                                          scenario="scenario_b_original_reference_transformed_probe",
                                          subsets=subsets)
        self.assertEqual(set(manifest.identity_ids), set(subsets.scoring_identities))
        self.assertEqual({p.first_role for p in manifest.pairs}, {"original_reference"})
        self.assertEqual({p.second_role for p in manifest.pairs}, {"transformed_probe"})
        manifest.validate(inventory, split, subsets)

    def test_insufficient_image_identities_are_recorded(self):
        inventory = synthetic_inventory()
        inventory["subject-12"] = ["image-12-only.png"]
        split = make_split(inventory)
        # Pick a partition that contains the singleton and at least two eligible identities.
        partition = split.identity_assignments["subject-12"]
        selected = split.partitions[partition]
        self.assertGreaterEqual(sum(len(inventory[i]) >= 2 for i in selected), 2)
        manifest = generate_pair_manifest(inventory, split, partition=partition, subset_role="partition",
                                          pair_count_per_class=1, seed=12,
                                          scenario="scenario_a_both_transformed")
        self.assertIn("subject-12", manifest.excluded_insufficient_images)
        manifest.validate(inventory, split)

    def test_duplicate_and_reversed_pairs_rejected(self):
        inventory = synthetic_inventory()
        split = make_split(inventory)
        manifest = generate_pair_manifest(inventory, split, partition="final_test", subset_role="partition",
                                          pair_count_per_class=2, seed=21,
                                          scenario="scenario_a_both_transformed")
        duplicated = replace(manifest, pairs=manifest.pairs + (manifest.pairs[0],))
        with self.assertRaises(ValueError):
            duplicated._validate_structure()
        pair = manifest.pairs[0]
        reversed_pair = replace(pair, first_image_id=pair.second_image_id, second_image_id=pair.first_image_id)
        reversed_manifest = replace(manifest, pairs=(pair, reversed_pair) + manifest.pairs[1:])
        with self.assertRaises(ValueError):
            reversed_manifest._validate_structure()

    def test_manifest_rejects_wrong_image_reference_and_tampered_json(self):
        inventory = synthetic_inventory()
        split = make_split(inventory)
        manifest = generate_pair_manifest(inventory, split, partition="final_test", subset_role="partition",
                                          pair_count_per_class=1, seed=7,
                                          scenario="scenario_a_both_transformed")
        broken = json.loads(manifest.to_json())
        broken["pairs"][0]["first_image_id"] = "not-in-inventory.png"
        with self.assertRaises(ValueError):
            PairManifest.from_json(json.dumps(broken))
        wrong_inventory = dict(inventory)
        wrong_inventory[manifest.identity_ids[0]] = ["substituted.png"]
        with self.assertRaises(ValueError):
            manifest.validate(wrong_inventory, split)

    def test_small_identity_sets_fail_clearly(self):
        inventory = synthetic_inventory(3)
        split = split_identities(inventory, dataset_identifier="small", dataset_release="r1", seed=1,
                                 counts={"development": 1, "optimization": 1, "validation": 1, "final_test": 0})
        with self.assertRaises(ValueError):
            generate_pair_manifest(inventory, split, partition="development", subset_role="partition",
                                   pair_count_per_class=1, seed=1, scenario="scenario_a_both_transformed")
        empty_subsets = split_calibration_scoring_identities(split, "final_test", seed=1, calibration_count=0)
        self.assertEqual(empty_subsets.calibration_identities, ())
        with self.assertRaises(ValueError):
            generate_pair_manifest(inventory, split, partition="final_test", subset_role="calibration",
                                   pair_count_per_class=1, seed=1,
                                   scenario="scenario_a_both_transformed", subsets=empty_subsets)

    def test_subsets_reject_overlap(self):
        with self.assertRaises(ValueError):
            EvaluationIdentitySubsets("0" * 64, "validation", 1, "counts", {},
                                      ("same",), ("same",))


if __name__ == "__main__":
    unittest.main()
