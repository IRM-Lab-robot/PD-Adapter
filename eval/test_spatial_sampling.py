"""Checks that distant proposals really occupy their promised spatial bands."""

import unittest

import numpy as np

from spatial_sampling import layer_slices, sample_stratified


class SpatialSamplingTest(unittest.TestCase):
    def test_shell_coverage_budget_seed_and_reproducibility(self):
        for dims in (12, 15):
            with self.subTest(dims=dims):
                means = np.zeros((2, dims), dtype=np.float32)
                means[1, :3] = (.11, -.07, .03)
                stds = np.full_like(means, .01)
                a = sample_stratified(means, stds, 288, np.random.default_rng(43))
                b = sample_stratified(means, stds, 288, np.random.default_rng(43))
                self.assertEqual(a.shape, (2, 288, dims))
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(a[:, 0], means)
                distance = np.linalg.norm(a[:, :, :3] - means[:, None, :3], axis=-1)
                self.assertTrue(np.all((distance[:, 96:192] >= .03 - 1e-6)
                                       & (distance[:, 96:192] <= .06 + 1e-6)))
                self.assertTrue(np.all((distance[:, 192:] >= .06 - 1e-6)
                                       & (distance[:, 192:] <= .12 + 1e-6)))
                self.assertTrue(np.all(np.isfinite(a)))

    def test_invalid_budget_rejected(self):
        with self.assertRaises(ValueError):
            sample_stratified(np.zeros((1, 12), dtype=np.float32),
                              np.ones((1, 12), dtype=np.float32), 2, np.random.default_rng(1))

    def test_32_candidates_cover_three_bands(self):
        means = np.zeros((2, 12), dtype=np.float32)
        stds = np.full_like(means, .01)
        samples = sample_stratified(means, stds, 32, np.random.default_rng(7))
        bands = layer_slices(32)
        self.assertEqual([band.stop - band.start for band in bands], [11, 11, 10])
        self.assertEqual([b.stop - b.start for b in layer_slices(288)], [96, 96, 96])
        radii = np.linalg.norm(samples[:, :, :3] - means[:, None, :3], axis=-1)
        for band, (lo, hi) in zip(bands[1:], ((.03, .06), (.06, .12))):
            self.assertTrue(np.all((radii[:, band] >= lo - 1e-6)
                                   & (radii[:, band] <= hi + 1e-6)))


if __name__ == "__main__":
    unittest.main()
