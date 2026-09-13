"""Lightweight model, public-checkpoint, and synthetic-audio regression tests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
import torch
from torch.nn import functional as F

from muyuset.checkpoints import load_models
from muyuset.diffusion import GeometryDiffusion, diffusion_loss, sample_ddim
from muyuset.frontend import (
    extract_signatures,
    load_audio,
    load_signatures,
    normalize_for_diffusion,
    validate_signatures,
)
from muyuset.renderer import ACTION_CODES
from muyuset.surrogate import ForwardSurrogate, predict_action_panel


ROOT = Path(__file__).resolve().parents[1]


def tiny_diffusion():
    return GeometryDiffusion(
        model_dim=16, heads=2, note_layers=1, slot_layers=1,
        diffusion_hidden_dim=32, time_dim=8, residual_blocks=1,
        diffusion_steps=8, dropout=0.0,
    ).eval()


class DiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(103)
        self.model = tiny_diffusion()
        self.inputs = {
            "features": torch.randn(2, 4, 3, 64),
            "strike_mask": torch.ones(2, 4, dtype=torch.bool),
        }

    def test_loss_is_velocity_prediction_and_backpropagates(self):
        target, noise = torch.randn(2, 24), torch.randn(2, 24)
        timestep = torch.tensor([0, 7])
        loss, details = diffusion_loss(
            self.model, target, self.inputs, noise=noise, timestep=timestep,
        )
        alpha = self.model.alpha_bars[timestep, None]
        expected = alpha.sqrt() * noise - (1 - alpha).sqrt() * target
        self.assertEqual(self.model.prediction_type, "v")
        torch.testing.assert_close(details["target_velocity"], expected)
        torch.testing.assert_close(loss, F.mse_loss(details["prediction"], expected))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))

    def test_ddim_shape_finite_and_seed_repeatability(self):
        first = sample_ddim(self.model, self.inputs, samples=3, steps=4, seed=71)
        again = sample_ddim(self.model, self.inputs, samples=3, steps=4, seed=71)
        other = sample_ddim(self.model, self.inputs, samples=3, steps=4, seed=72)
        self.assertEqual(tuple(first.shape), (2, 3, 24))
        self.assertTrue(torch.isfinite(first).all().item())
        self.assertFalse(first.requires_grad)
        self.assertTrue(torch.equal(first, again))
        self.assertFalse(torch.equal(first, other))

    def test_masked_nan_padding_does_not_change_condition_or_samples(self):
        padded = {
            "features": torch.cat([
                self.inputs["features"], torch.full((2, 3, 3, 64), float("nan")),
            ], dim=1),
            "strike_mask": torch.cat([
                self.inputs["strike_mask"], torch.zeros(2, 3, dtype=torch.bool),
            ], dim=1),
        }
        with torch.no_grad():
            original = self.model.encode_condition(**self.inputs)
            masked = self.model.encode_condition(**padded)
        self.assertTrue(torch.isfinite(masked).all().item())
        torch.testing.assert_close(original, masked, rtol=1e-5, atol=2e-6)
        first = sample_ddim(self.model, self.inputs, samples=2, steps=3, seed=3)
        second = sample_ddim(self.model, padded, samples=2, steps=3, seed=3)
        torch.testing.assert_close(first, second, rtol=1e-5, atol=2e-6)

    def test_invalid_model_inputs_and_sampling_options_are_rejected(self):
        bad_inputs = [
            {**self.inputs, "actions": torch.zeros(2, 4, 3)},
            {**self.inputs, "features": torch.zeros(2, 4, 2, 64)},
            {**self.inputs, "strike_mask": torch.zeros(2, 4, dtype=torch.bool)},
            {**self.inputs, "strike_mask": torch.ones(2, 3, dtype=torch.bool)},
        ]
        for inputs in bad_inputs:
            with self.subTest(keys=tuple(inputs), shape=inputs["features"].shape):
                with self.assertRaises(ValueError):
                    sample_ddim(self.model, inputs, samples=1, steps=2, seed=1)
        for kwargs in ({"samples": 0}, {"steps": 9},
                       {"seed": 1, "generator": torch.Generator()}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    sample_ddim(self.model, self.inputs, **kwargs)


class PublicCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.diffusion_path = ROOT / "weights" / "diffusion.pt"
        cls.surrogate_path = ROOT / "weights" / "surrogate.pt"
        cls.diffusion, cls.surrogate, cls.metadata = load_models(
            cls.diffusion_path, cls.surrogate_path, device="cpu",
        )

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_both_public_checkpoints_load_with_matching_statistics(self):
        self.assertIsInstance(self.diffusion, GeometryDiffusion)
        self.assertIsInstance(self.surrogate, ForwardSurrogate)
        forward_metadata = torch.load(
            self.surrogate_path, map_location="cpu", weights_only=True,
        )
        for name in ("latent_mean", "latent_std"):
            np.testing.assert_array_equal(
                np.asarray(self.metadata[name]), np.asarray(forward_metadata[name]),
            )
            self.assertEqual(np.asarray(self.metadata[name]).shape, (8,))
        self.assertEqual(self.diffusion.conditioner.model_dim, 192)
        self.assertEqual(self.diffusion.diffusion_steps, 1000)
        self.assertEqual(self.surrogate.model_dim, 384)
        self.assertFalse(self.diffusion.training)
        self.assertFalse(self.surrogate.training)
        for model in (self.diffusion, self.surrogate):
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_public_surrogate_has_finite_geometry_gradients_but_frozen_weights(self):
        geometry = torch.zeros(1, 3, 8, requires_grad=True)
        panel = predict_action_panel(
            self.surrogate, geometry, torch.tensor(ACTION_CODES.copy(), dtype=torch.long),
        )
        self.assertEqual(tuple(panel.shape), (1, 3, 45, 64))
        self.assertTrue(torch.isfinite(panel).all().item())
        torch.testing.assert_close(
            panel.mean(dim=-1), torch.zeros(1, 3, 45), atol=2e-6, rtol=0,
        )
        panel.square().mean().backward()
        self.assertIsNotNone(geometry.grad)
        self.assertTrue(torch.isfinite(geometry.grad).all().item())
        self.assertGreater(geometry.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.surrogate.parameters()))

    def test_checkpoint_loader_rejects_missing_model_state_key_strictly(self):
        first = torch.load(self.diffusion_path, map_location="cpu", weights_only=True)
        second = torch.load(self.surrogate_path, map_location="cpu", weights_only=True)
        first["model_state"] = dict(first["model_state"])
        first["model_state"].pop(next(iter(first["model_state"])))
        with patch("muyuset.checkpoints.torch.load", side_effect=[first, second]):
            with self.assertRaises(RuntimeError):
                load_models(self.diffusion_path, self.surrogate_path)

    def test_checkpoint_loader_rejects_differing_latent_statistics(self):
        first = torch.load(self.diffusion_path, map_location="cpu", weights_only=True)
        second = torch.load(self.surrogate_path, map_location="cpu", weights_only=True)
        second["latent_mean"] = np.asarray(second["latent_mean"]).copy()
        second["latent_mean"][0] += 0.1
        with patch("muyuset.checkpoints.torch.load", side_effect=[first, second]):
            with self.assertRaisesRegex(ValueError, "statistics must match"):
                load_models(self.diffusion_path, self.surrogate_path)


class FrontendTests(unittest.TestCase):
    def test_gain_alignment_and_standardization(self):
        mean = np.arange(192, dtype=np.float32).reshape(3, 64) / 10
        std = np.full((3, 64), 2.0, dtype=np.float32)
        std[0, 0] = 0  # Normalization intentionally clamps tiny standard deviations.
        signatures = np.broadcast_to(mean, (12, 3, 64)).copy()
        signatures[:, 1] += 4
        shifts = np.arange(12, dtype=np.float32)[:, None, None]
        signatures += shifts
        result = normalize_for_diffusion(
            signatures, {"feature_mean": mean, "feature_std": std},
        )
        expected = np.zeros_like(signatures)
        expected[:, 1] = 2
        np.testing.assert_allclose(result, expected, atol=1e-3, rtol=0)
        self.assertTrue(result.flags.c_contiguous)

    def test_bad_features_are_rejected(self):
        bad = [np.zeros((11, 3, 64)), np.zeros((12, 2, 64)),
               np.zeros((12, 3, 63)), np.zeros((12, 3, 64, 1)),
               np.full((12, 3, 64), np.nan), np.full((12, 3, 64), np.inf)]
        for values in bad:
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    validate_signatures(values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npy"
            np.save(path, np.zeros((11, 3, 64)))
            with self.assertRaises(ValueError):
                load_signatures(path)

    def test_signature_extraction_rejects_invalid_waveforms_and_onsets(self):
        samples = np.zeros(4410)
        valid_onsets = np.linspace(0.005, 0.09, 12)
        for wave, onsets, rate in (
            (samples, np.r_[-0.001, valid_onsets], 44100),
            (samples, np.r_[valid_onsets, 0.1], 44100),
            (samples, np.full(12, np.nan), 44100),
            (samples, valid_onsets.reshape(3, 4), 44100),
            (samples, valid_onsets, 22050),
            (np.zeros((4410, 2)), valid_onsets, 44100),
            (np.full(4410, np.nan), valid_onsets, 44100),
            (np.zeros(0), valid_onsets, 44100),
        ):
            with self.subTest(rate=rate, waveform_shape=wave.shape):
                with self.assertRaises(ValueError):
                    extract_signatures(wave, onsets, rate)

    def test_real_frontend_with_temporary_zero_wav_and_onset_json(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path, plan_path = Path(directory) / "zero.wav", Path(directory) / "onsets.json"
            wavfile.write(wav_path, 44100, np.zeros(8820, dtype=np.int16))
            onsets = np.linspace(0.005, 0.18, 12)
            plan_path.write_text(json.dumps({"notes": [
                {"onset_s": float(value)} for value in onsets
            ]}), encoding="utf-8")
            signatures, returned = load_audio(wav_path, plan_path)
            self.assertEqual(signatures.shape, (12, 3, 64))
            self.assertEqual(signatures.dtype, np.float32)
            self.assertTrue(np.isfinite(signatures).all())
            self.assertTrue(np.all(signatures >= -70))
            np.testing.assert_array_equal(returned, onsets)

    def test_wav_and_plan_input_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path, plan_path = Path(directory) / "input.wav", Path(directory) / "plan.json"
            onsets = np.linspace(0.005, 0.18, 12)
            good_notes = [{"time_s": float(value)} for value in onsets]
            plan_path.write_text(json.dumps({"performance_plan": good_notes}), encoding="utf-8")
            for rate, samples in ((22050, np.zeros(4410, dtype=np.int16)),
                                  (44100, np.zeros((8820, 2), dtype=np.int16)),
                                  (44100, np.full(8820, np.nan, dtype=np.float32))):
                wavfile.write(wav_path, rate, samples)
                with self.subTest(rate=rate, shape=samples.shape):
                    with self.assertRaises(ValueError):
                        load_audio(wav_path, plan_path)
            wavfile.write(wav_path, 44100, np.zeros(8820, dtype=np.int16))
            for notes in ([], [{}], [None], good_notes[::-1],
                          good_notes + [good_notes[-1]],
                          [{"time_s": float("nan")}]):
                plan_path.write_text(json.dumps({"notes": notes}), encoding="utf-8")
                with self.subTest(notes=notes):
                    with self.assertRaises(ValueError):
                        load_audio(wav_path, plan_path)
            plan_path.write_text(json.dumps({"notes": good_notes}), encoding="utf-8")
            for kwargs in ({"start": -1}, {"end": 0.3}, {"start": 0.1, "end": 0.1},
                           {"start": float("nan")}):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaises(ValueError):
                        load_audio(wav_path, plan_path, **kwargs)

    def test_plan_rejects_out_of_waveform_onsets_even_with_enough_valid_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path, plan_path = Path(directory) / "input.wav", Path(directory) / "plan.json"
            wavfile.write(wav_path, 44100, np.zeros(8820, dtype=np.int16))
            good = [{"onset_s": float(t)} for t in np.linspace(0.005, 0.18, 12)]
            for notes in ([{"onset_s": -0.01}] + good, good + [{"onset_s": 0.25}]):
                plan_path.write_text(json.dumps({"notes": notes}), encoding="utf-8")
                with self.subTest(outside=notes[0] if notes[0]["onset_s"] < 0 else notes[-1]):
                    with self.assertRaises(ValueError):
                        load_audio(wav_path, plan_path)


if __name__ == "__main__":
    unittest.main()
