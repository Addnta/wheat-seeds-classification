"""Pengujian fondasi loader, preprocessing, model, evaluasi, dan prediksi."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import CLASS_MAP, FEATURES, LABEL_ORDER, dataset_diagnostics, load_dataset
from src.evaluation import best_model, evaluate_models
from src.models import MODEL_NAMES, build_models
from src.prediction import fit_final_models, predict_sample, validate_input
from src.preprocessing import split_features_target


class FoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()
        cls.evaluation = evaluate_models(cls.data)
        cls.final_models = fit_final_models(cls.data, cls.evaluation)

    def test_dataset_shape_labels_and_quality(self):
        self.assertEqual(self.data.shape, (210, 9))
        self.assertEqual(tuple(self.data.loc[:, FEATURES].columns), FEATURES)
        self.assertTrue(self.data["class_id"].isin(CLASS_MAP).all())
        self.assertEqual(
            self.data["variety"].value_counts().reindex(LABEL_ORDER).tolist(),
            [70, 70, 70],
        )
        diagnostic = dataset_diagnostics(self.data)
        self.assertEqual(diagnostic["rows"], 210)
        self.assertEqual(diagnostic["duplicate_rows"], 0)
        self.assertEqual(int(diagnostic["missing_values"].sum()), 0)

    def test_preprocessing_excludes_target_from_features(self):
        X, y = split_features_target(self.data)
        self.assertEqual(list(X.columns), list(FEATURES))
        self.assertEqual(X.shape, (210, 7))
        self.assertEqual(y.name, "variety")
        self.assertNotIn("class_id", X.columns)
        self.assertNotIn("variety", X.columns)

    def test_models_match_notebook_baseline(self):
        models = build_models()
        self.assertEqual(set(models), set(MODEL_NAMES))
        self.assertIsInstance(models["knn"], Pipeline)
        self.assertIsInstance(models["knn"].named_steps["scaler"], StandardScaler)
        self.assertEqual(models["knn"].named_steps["classifier"].n_neighbors, 5)
        self.assertEqual(models["knn"].named_steps["classifier"].p, 2)
        self.assertNotIsInstance(models["decision_tree"], Pipeline)
        self.assertEqual(models["decision_tree"].criterion, "gini")
        self.assertIsNone(models["decision_tree"].max_depth)
        self.assertEqual(models["decision_tree"].random_state, 42)
        self.assertEqual(models["gaussian_nb"].var_smoothing, 1e-9)

    def test_cross_validation_and_oof_are_separate_outputs(self):
        result = self.evaluation
        self.assertEqual(len(result.cv_summary), 3)
        self.assertEqual(set(result.cv_summary["model_key"]), set(MODEL_NAMES))
        for key in MODEL_NAMES:
            self.assertEqual(len(result.cv_fold_scores[key]["accuracy"]), 5)
            self.assertTrue(0 <= result.cv_summary.loc[
                result.cv_summary.model_key == key, "accuracy_mean"
            ].iloc[0] <= 1)
            self.assertEqual(int(result.confusion_matrices[key].to_numpy().sum()), 210)
            self.assertEqual(len(result.oof_predictions[key]), 210)
            self.assertIn("f1_macro", result.oof_metrics[key])
        self.assertIn(best_model(result, "accuracy"), MODEL_NAMES)
        self.assertIn(best_model(result, "f1_macro"), MODEL_NAMES)
        with self.assertRaises(ValueError):
            best_model(result, "precision_macro")

    def test_final_training_requires_matching_completed_evaluation(self):
        changed = self.data.copy()
        changed.loc[0, "area"] += 0.01
        with self.assertRaises(ValueError):
            fit_final_models(changed, self.evaluation)

    def test_prediction_validation_probabilities_and_range_warning(self):
        example = self.data.loc[0, FEATURES].astype(float).tolist()
        result = predict_sample(self.final_models, self.data, "knn", example)
        self.assertIn(result.class_name, LABEL_ORDER)
        self.assertIn(result.class_id, CLASS_MAP)
        self.assertIsNotNone(result.probabilities)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=7)

        outside = list(example)
        outside[0] = float(self.data["area"].max() + 10)
        result_outside = predict_sample(self.final_models, self.data, "knn", outside)
        self.assertIn("area", result_outside.out_of_range_features)
        self.assertEqual(validate_input(example).shape, (7,))
        for invalid in ([1, 2], [1, 2, 3, 4, 5, 6, np.nan], [1, 2, 3, 4, 5, 6, np.inf]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_input(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
