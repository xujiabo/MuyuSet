"""Cheap budget/selection regressions plus a tiny real surrogate optimizer."""

from dataclasses import replace
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

import muyuset.pipeline as pipeline
from muyuset.refinement import objective, refine
from muyuset.renderer import ACTION_CODES, OOD_LATENT_LOWER, OOD_LATENT_UPPER, Score
from muyuset.surrogate import ForwardSurrogate, predict_action_panel


class PipelineTests(unittest.TestCase):
    def mocked_run(self, original_specs=None, refined_specs=None, mutate_initial=False):
        """Run public orchestration without checkpoint, physics, or Adam work."""
        config = pipeline.Config()
        count = config.subsets * config.samples
        proposals = np.arange(count * 24, dtype=np.float32).reshape(6, 16, 24) / 10000
        original_copy = proposals.reshape(count, 3, 8).copy()
        metadata = {
            "feature_mean": np.zeros((3, 64), dtype=np.float32),
            "feature_std": np.ones((3, 64), dtype=np.float32),
            "latent_mean": np.full(8, .1, dtype=np.float32),
            "latent_std": np.full(8, 2, dtype=np.float32),
        }
        feasible_usage = (16, 16, 16)
        original_specs = original_specs or [(1 if i == 0 else 10 + i, feasible_usage, True)
                                            for i in range(count)]
        refined_specs = refined_specs or [(100 + i, feasible_usage, True)
                                          for i in range(config.tracks)]
        specs = iter(original_specs + refined_specs)
        scored_raw = []

        def exact(raw, observed):
            value, usage, valid = next(specs)
            scored_raw.append(np.array(raw, copy=True))
            return Score(float(value), np.asarray(usage, dtype=np.int32), bool(valid),
                         None if valid else "synthetic invalid geometry",
                         np.zeros(48, dtype=np.int16), np.zeros(48, dtype=np.int16),
                         np.array(raw, copy=True))

        def fake_refine(model, initial, observed, actions, lower, upper, steps, rate):
            self.assertEqual(initial.shape, (32, 3, 8))
            self.assertEqual(observed.shape, (48, 64))
            np.testing.assert_array_equal(actions, ACTION_CODES)
            np.testing.assert_allclose(lower, (OOD_LATENT_LOWER - .1) / 2)
            np.testing.assert_allclose(upper, (OOD_LATENT_UPPER - .1) / 2)
            self.assertEqual(steps, 200)
            self.assertEqual(rate, .025)
            if mutate_initial:
                initial[:] = -123  # Elite advanced indexing must protect originals.
            return (np.full((32, 3, 8), -.25, dtype=np.float32),
                    np.zeros(32, dtype=np.int64), np.zeros((201, 32), dtype=np.float32))

        with patch.object(pipeline, "load_models", return_value=(object(), object(), metadata)), \
                patch.object(pipeline, "sample_ddim", return_value=torch.from_numpy(proposals)) as sample, \
                patch.object(pipeline, "evaluate", side_effect=exact) as evaluate, \
                patch.object(pipeline, "refine", side_effect=fake_refine) as optimize, \
                patch.object(pipeline, "geometry_parameters", return_value=[{}, {}, {}]):
            result, arrays = pipeline.run(np.zeros((48, 3, 64), dtype=np.float32),
                                          "unused-diffusion", "unused-surrogate", config,
                                          progress=lambda _: None)
        return result, arrays, original_copy, scored_raw, sample, evaluate, optimize

    def test_default_exact_budget_and_surrogate_updates(self):
        result, arrays, _, _, sample, evaluate, optimize = self.mocked_run()
        self.assertEqual(result["exact_candidate_attempts"], 128)
        self.assertEqual(result["surrogate_updates"], 6400)
        self.assertTrue(result["paper_budget_settings"])
        self.assertEqual(evaluate.call_count, 128)
        sample.assert_called_once()
        optimize.assert_called_once()
        self.assertEqual(sample.call_args.kwargs["samples"], 16)
        self.assertEqual(sample.call_args.kwargs["steps"], 48)
        self.assertEqual(sample.call_args.args[1]["features"].shape, (6, 12, 3, 64))
        self.assertEqual(arrays["originals_standardized"].shape, (96, 3, 8))
        self.assertEqual(arrays["refined_standardized"].shape, (32, 3, 8))
        self.assertEqual(arrays["surrogate_objective_history"].shape, (201, 32))
        self.assertEqual(result["minimum_slot_usage"], 5)

    def test_originals_are_preserved_and_cannot_regress(self):
        result, arrays, original, scored_raw, *_ = self.mocked_run(mutate_initial=True)
        np.testing.assert_array_equal(arrays["originals_standardized"], original)
        np.testing.assert_allclose(np.asarray(scored_raw[:96]), original * 2 + .1)
        self.assertEqual(result["best"]["source"], "original")
        self.assertEqual(result["best"]["candidate_index"], 0)
        self.assertEqual(result["best"]["objective"], 1)

    def test_exact_ties_prefer_original(self):
        ties = [(1, (16, 16, 16), True)] * 32
        result, *_ = self.mocked_run(refined_specs=ties)
        self.assertEqual(result["best"]["source"], "original")
        self.assertEqual(result["best"]["candidate_index"], 0)

    def test_better_refinement_can_win(self):
        refined = [(.25, (16, 16, 16), True)] + [(100, (16, 16, 16), True)] * 31
        result, *_ = self.mocked_run(refined_specs=refined)
        self.assertEqual(result["best"]["source"], "refined")
        self.assertEqual(result["best"]["candidate_index"], 96)
        self.assertEqual(result["best"]["objective"], .25)

    def test_elites_and_final_choice_are_feasible_first(self):
        originals = [(0, (48, 0, 0), True), (.1, (48, 0, 0), False)]
        originals += [(10 + i, (16, 16, 16), True) for i in range(94)]
        result, arrays, *_ = self.mocked_run(original_specs=originals)
        np.testing.assert_array_equal(arrays["elite_indices"], np.arange(2, 34))
        self.assertEqual(result["best"]["candidate_index"], 2)
        self.assertEqual(result["best"]["objective"], 10)
        self.assertIsNone(result["candidate_scores"][1]["objective"])
        json.dumps(result, allow_nan=False)

    def test_no_feasible_candidate_has_no_fake_best(self):
        infeasible = [(1, (48, 0, 0), True)]
        result, *_ = self.mocked_run(original_specs=infeasible * 96,
                                     refined_specs=infeasible * 32)
        self.assertEqual(result["status"], "no_feasible_candidate")
        self.assertEqual(result["feasible_candidates"], 0)
        self.assertIsNone(result["best"])
        json.dumps(result, allow_nan=False)

    def test_all_invalid_candidates_are_json_safe(self):
        invalid = [(float("inf"), (0, 0, 0), False)]
        result, *_ = self.mocked_run(original_specs=invalid * 96,
                                     refined_specs=invalid * 32)
        self.assertEqual(result["renderable_candidates"], 0)
        self.assertIsNone(result["best"])
        self.assertTrue(all(s["objective"] is None for s in result["candidate_scores"]))
        json.dumps(result, allow_nan=False)

    def test_config_rejects_invalid_inputs(self):
        for field in ("subsets", "samples", "ddim_steps", "tracks", "steps", "workers"):
            for invalid in (0, -1, True, 1.5):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    replace(pipeline.Config(), **{field: invalid}).validate()
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.subTest(learning_rate=invalid), self.assertRaises(ValueError):
                replace(pipeline.Config(), learning_rate=invalid).validate()
        for invalid in (-1, 2**32, True, 1.5):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                replace(pipeline.Config(), seed=invalid).validate()
        with self.assertRaises(ValueError):
            replace(pipeline.Config(), tracks=97).validate()
        pipeline.Config().validate()


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_48_hit_soft_usage_target_is_five_over_48(self):
        # Slot 0 has usage .102: above .1 but below 5/48. The old fixed .1
        # formula incurs zero penalty and must fail this analytic regression.
        probabilities = torch.tensor([.102, .449, .449], dtype=torch.float64)
        costs = 1 - probabilities.log()
        panel = costs[None, :, None, None].expand(1, 3, 45, 64)
        observed = torch.zeros(48, 64, dtype=torch.float64)
        base = 1.15 * (1 + math.log(3))
        expected_penalty = 100 * (5 / 48 - .102)**2
        actual = objective(panel, observed).item()
        self.assertAlmostEqual(actual, base + expected_penalty, places=11)
        self.assertGreater(actual - base, 1e-4)

    def test_tiny_real_surrogate_refinement_is_finite_bounded_and_retained(self):
        torch.manual_seed(91)
        model = ForwardSurrogate(model_dim=16, residual_hidden_dim=24,
                                 residual_blocks=1, action_embedding_dim=4,
                                 fourier_frequencies=(.5, 1.)).train()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        initial = np.linspace(-2, 2, 3 * 3 * 8, dtype=np.float32).reshape(3, 3, 8)
        unchanged = initial.copy()
        observed = np.random.RandomState(7).normal(size=(48, 64)).astype(np.float32)
        lower = np.full(8, -.2, dtype=np.float32)
        upper = np.full(8, .3, dtype=np.float32)
        best, best_steps, history = refine(model, initial, observed, ACTION_CODES,
                                          lower, upper, steps=8, learning_rate=.08)
        np.testing.assert_array_equal(initial, unchanged)
        self.assertEqual(best.shape, (3, 3, 8))
        self.assertEqual(history.shape, (9, 3))
        self.assertTrue(np.isfinite(best).all() and np.isfinite(history).all())
        self.assertTrue(np.all(best >= lower) and np.all(best <= upper))
        np.testing.assert_array_equal(best_steps, history.argmin(axis=0))
        with torch.no_grad():
            retained = objective(predict_action_panel(model, torch.from_numpy(best),
                                 torch.tensor(ACTION_CODES.astype(np.int64))),
                                 torch.from_numpy(observed)).numpy()
        np.testing.assert_allclose(retained, history.min(axis=0), rtol=1e-6, atol=1e-6)
        self.assertTrue(np.all(history.min(axis=0) <= history[0]))
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
