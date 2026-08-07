"""Regression tests for core leakage and evaluation utilities."""

import unittest

import numpy as np
import pandas as pd

from src.build_temporal_graph import patient_token, validate_temporal_edges
from src.create_splits import patient_split
from src.evaluate_locked_temporal_model import paired_patient_bootstrap
from src.train_baseline import diagnosis_group, metric_bundle
from src.train_graphsage_prototype import neighbour_matrix
from src.train_temporal_baseline import add_causal_history


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

    def test_causal_history_reads_only_previous_encounter(self):
        frame = pd.DataFrame(
            {
                "patient_nbr": ["a", "a", "b"],
                "encounter_id": [10, 20, 15],
                "time_in_hospital": [2, 7, 4],
                "num_lab_procedures": [10, 20, 30],
                "num_medications": [1, 2, 3],
                "number_diagnoses": [2, 3, 4],
                "diag_1_group": ["Diabetes", "Circulatory", "Other"],
                "A1Cresult": ["None", ">8", "Norm"],
                "insulin": ["No", "Up", "Steady"],
                "change": ["No", "Ch", "No"],
            }
        )
        result = add_causal_history(frame).set_index("encounter_id")
        self.assertEqual(result.loc[20, "previous_time_in_hospital"], 2)
        self.assertEqual(result.loc[20, "previous_diag_1_group"], "Diabetes")
        self.assertEqual(result.loc[10, "previous_diag_1_group"], "No_prior_encounter")
        self.assertEqual(result.loc[15, "observed_prior_encounters"], 0)

    def test_neighbour_matrix_never_uses_future_row(self):
        matrix = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        neighbours = neighbour_matrix(matrix, np.array([-1, 0, 1]))
        np.testing.assert_array_equal(neighbours[:, 0], np.array([0.0, 1.0, 2.0]))

    def test_temporal_edge_validator_rejects_reversed_edge(self):
        nodes = pd.DataFrame({"node_id": ["10", "20"]})
        labels = pd.DataFrame({"node_id": ["10", "20"], "target_30d": [0, 1]})
        valid = pd.DataFrame({"source_current": ["20"], "target_previous": ["10"]})
        validate_temporal_edges(nodes, labels, valid)
        reversed_edge = pd.DataFrame(
            {"source_current": ["10"], "target_previous": ["20"]}
        )
        with self.assertRaises(AssertionError):
            validate_temporal_edges(nodes, labels, reversed_edge)

    def test_paired_bootstrap_is_deterministic(self):
        y = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
        patients = np.array(["a", "a", "b", "c", "d", "e"])
        reference = np.array([0.2, 0.6, 0.4, 0.55, 0.3, 0.7])
        candidate = np.array([0.1, 0.8, 0.2, 0.75, 0.15, 0.9])
        first = paired_patient_bootstrap(
            y, reference, candidate, patients, repeats=20
        )
        second = paired_patient_bootstrap(
            y, reference, candidate, patients, repeats=20
        )
        self.assertEqual(first, second)
        self.assertGreater(
            first["brier_score"]["probability_candidate_better"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
