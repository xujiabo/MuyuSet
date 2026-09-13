"""Frozen renderer contract tests; one real 135-signature panel is shared."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np

import muyuset.renderer as renderer


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = np.zeros((3, 8), dtype=np.float64)
        cls.raw[:, 0] = (0.05, 0.0, -0.05)
        cls.ordered = renderer.canonicalize(cls.raw)
        # Only this call executes the full exact audible backend.
        cls.signatures = renderer.render_signatures(cls.raw)
        cls.panel = renderer.prepare_attack_shapes(cls.signatures)
        cls.expected_slots = np.repeat(np.arange(3), 3).astype(np.int16)
        cls.expected_actions = np.tile((0, 16, 44), 3).astype(np.int16)
        cls.observed = cls.panel[cls.expected_slots, cls.expected_actions].copy()

    def score(self, objective, usage, valid=True):
        return renderer.Score(
            objective=float(objective), usage=np.asarray(usage, dtype=np.int32),
            valid=valid, error=None, slots=np.zeros(1, dtype=np.int16),
            action_indices=np.zeros(1, dtype=np.int16), raw_latents=self.ordered.copy(),
        )

    def test_real_signature_shape_and_action_order(self):
        self.assertEqual(self.signatures.shape, (3, 45, 3, 64))
        self.assertEqual(self.signatures.dtype, np.float32)
        self.assertTrue(np.isfinite(self.signatures).all())
        expected = np.asarray(list(itertools.product(range(3), range(3), range(5))))
        np.testing.assert_array_equal(renderer.ACTION_CODES, expected)
        self.assertFalse(renderer.ACTION_CODES.flags.writeable)
        conditions = renderer.build_conditions()
        actual = [(c.material_index, c.velocity_index, c.position_index) for c in conditions]
        np.testing.assert_array_equal(actual, expected)

    def test_attack_floor_then_mean_center(self):
        absolute = self.signatures[..., 0, :].copy()
        expected = np.maximum(absolute, np.float32(-70.0))
        expected -= expected.mean(axis=-1, keepdims=True, dtype=np.float32)
        np.testing.assert_array_equal(self.panel, expected)
        np.testing.assert_allclose(self.panel.mean(axis=-1), 0.0, atol=1e-5)
        np.testing.assert_array_equal(self.signatures[..., 0, :], absolute)

    def test_canonicalization_is_permutation_invariant_and_idempotent(self):
        unchanged = self.raw.copy()
        for permutation in itertools.permutations(range(3)):
            np.testing.assert_array_equal(
                renderer.canonicalize(self.raw[list(permutation)]), self.ordered,
            )
        np.testing.assert_array_equal(renderer.canonicalize(self.ordered), self.ordered)
        np.testing.assert_array_equal(self.raw, unchanged)
        frequencies = [
            renderer.contact_modal_palette(renderer._object(row).parameters)[0][0]
            for row in self.ordered
        ]
        self.assertEqual(frequencies, sorted(frequencies))

    def test_canonicalization_uses_deterministic_raw_tie_break(self):
        # Artificial equal pitches exercise tie handling independently of the
        # physical parameter mapping; lexical raw ordering must be consistent.
        expected = np.asarray(sorted(self.raw.tolist()))
        with patch.object(renderer, "contact_modal_palette",
                          return_value=(np.asarray((155.0, 177.0)), np.ones(2))):
            for permutation in itertools.permutations(range(3)):
                np.testing.assert_array_equal(
                    renderer.canonicalize(self.raw[list(permutation)]), expected,
                )
            repeated = np.repeat(self.raw[:1], 3, axis=0)
            np.testing.assert_array_equal(renderer.canonicalize(repeated), repeated)

    def test_self_generated_events_have_zero_score_and_exact_labels(self):
        # Reuse the real backend panel rather than rendering the same bodies.
        with patch.object(renderer, "render_signatures", return_value=self.signatures):
            result = renderer.evaluate(self.raw[[2, 0, 1]], self.observed)
        self.assertTrue(result.valid, result.error)
        self.assertIsNone(result.error)
        self.assertEqual(result.objective, 0.0)
        np.testing.assert_array_equal(result.slots, self.expected_slots)
        np.testing.assert_array_equal(result.action_indices, self.expected_actions)
        np.testing.assert_array_equal(result.usage, (3, 3, 3))
        np.testing.assert_array_equal(result.raw_latents, self.ordered)

    def test_trimmed_mean_plus_p90_metric(self):
        # The same cached real action with constant offsets gives known RMS
        # costs while retaining the actual exact action-enumeration code.
        offsets = np.arange(1, 11, dtype=np.float32)
        observed = self.panel[0, 0][None, :] + offsets[:, None]
        with patch.object(renderer, "render_signatures", return_value=self.signatures):
            result = renderer.evaluate(self.raw, observed)
        self.assertTrue(result.valid, result.error)
        keep = len(offsets) - int(np.floor(0.1 * len(offsets)))
        expected = float(np.mean(offsets[:keep])) + 0.15 * float(np.percentile(offsets, 90))
        self.assertAlmostEqual(result.objective, expected, places=5)

    def test_hard_usage_feasible_first_deficit_and_invalid_last(self):
        feasible = self.score(100.0, (2, 2, 2))
        small_deficit = self.score(10.0, (1, 2, 3))
        big_deficit = self.score(0.1, (0, 2, 4))
        invalid = self.score(0.0, (100, 100, 100), valid=False)
        self.assertTrue(feasible.feasible(2))
        self.assertFalse(small_deficit.feasible(2))
        self.assertFalse(invalid.feasible(2))
        self.assertLess(feasible.rank(2), small_deficit.rank(2))
        self.assertLess(small_deficit.rank(2), big_deficit.rank(2))
        self.assertLess(big_deficit.rank(2), invalid.rank(2))
        self.assertLess(self.score(1.0, (1, 2, 3)).rank(2), small_deficit.rank(2))
        self.assertEqual(invalid.rank(2), (2, float("inf"), float("inf")))
        nonfinite = self.score(float("nan"), (2, 2, 2))
        self.assertFalse(nonfinite.feasible(2))
        self.assertEqual(nonfinite.rank(2)[0], 2)
        for minimum in (-1, 1.5, True):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                feasible.rank(minimum)

    def test_illegal_geometry_returns_invalid_without_rendering(self):
        illegal = self.raw.copy()
        illegal[0, 3:6] = np.log(100.0)  # walls exceed outer dimensions
        with patch.object(renderer, "render_signatures") as render:
            result = renderer.evaluate(illegal, self.observed)
        render.assert_not_called()
        self.assertFalse(result.valid)
        self.assertTrue(np.isinf(result.objective))
        self.assertIn("ValueError", result.error)
        self.assertEqual(result.raw_latents.shape, (3, 8))
        np.testing.assert_array_equal(result.usage, (0, 0, 0))
        np.testing.assert_array_equal(result.slots, np.full(9, -1))
        np.testing.assert_array_equal(result.action_indices, np.full(9, -1))

    def test_numerical_renderer_failure_is_invalid(self):
        with patch.object(renderer, "render_signatures",
                          side_effect=FloatingPointError("numerical failure")):
            result = renderer.evaluate(self.raw, self.observed)
        self.assertFalse(result.valid)
        self.assertIn("numerical failure", result.error)

    def test_systemic_runtime_error_is_not_swallowed(self):
        with patch.object(renderer, "render_signatures",
                          side_effect=RuntimeError("unexpected backend shape")):
            with self.assertRaisesRegex(RuntimeError, "unexpected backend shape"):
                renderer.evaluate(self.raw, self.observed)

    def test_geometry_parameters_are_canonical_si_fields(self):
        geometries = renderer.geometry_parameters(self.raw[[1, 2, 0]])
        self.assertEqual(len(geometries), 3)
        required = {
            "outer_length_m", "outer_depth_m", "outer_height_m",
            "cavity_length_m", "cavity_depth_m", "cavity_height_m",
            "mouth_width_m", "mouth_height_m", "minimum_wall_m",
            "cavity_offset_y_m", "cavity_offset_z_m", "mouth_center_z_m",
        }
        for row, geometry in zip(self.ordered, geometries):
            self.assertTrue(required.issubset(geometry))
            self.assertEqual(geometry, renderer.decode_latent(row).to_dict())
            self.assertAlmostEqual(geometry["outer_length_m"], 0.390 * np.exp(row[0]))
            self.assertAlmostEqual(geometry["outer_depth_m"], 0.370)
            self.assertAlmostEqual(geometry["outer_height_m"], 0.340)
            self.assertAlmostEqual(geometry["mouth_height_m"], 0.030)
            self.assertAlmostEqual(geometry["minimum_wall_m"], 0.006)
            self.assertLess(geometry["cavity_length_m"], geometry["outer_length_m"])


if __name__ == "__main__":
    unittest.main()
