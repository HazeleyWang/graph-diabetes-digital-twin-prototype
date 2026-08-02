"""Regression tests for core leakage and evaluation utilities."""

import unittest

import numpy as np

from src.build_temporal_graph import patient_token
from src.create_splits import patient_split
from src.train_baseline import diagnosis_group, metric_bundle


class CorePipelineTests(unittest.TestCase):
    def test_patient_split_is_deterministic_and_valid(self):
        first = patient_split("123456")
        self.assertEqual(first, patient_split("123456"))
        self.assertIn(first, {"train", "validation", "test"})

    def test_patient_token_is_pseudonymous_and_stable(self):
        token = patient_token("123456")
        self.assertEqual(token, patient_token("123456"))
        self.assertNotIn("123456", token)
        self.assertEqual(len(token), 16)

    def test_diagnosis_groups(self):
        self.assertEqual(diagnosis_group("250.83"), "Diabetes")
        self.assertEqual(diagnosis_group("414"), "Circulatory")
        self.assertEqual(diagnosis_group("?"), "Missing")
        self.assertEqual(diagnosis_group("V45"), "Supplementary")

    def test_metric_bundle_perfect_ranking(self):
        outcome = np.array([0, 0, 1, 1], dtype=np.float32)
        probability = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        result = metric_bundle(outcome, probability)
        self.assertAlmostEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["average_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()

