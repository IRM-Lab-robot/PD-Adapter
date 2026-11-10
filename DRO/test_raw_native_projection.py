"""Raw actively drives passive joints; native also uses passive source data."""
import unittest

import numpy as np
import torch

from DRO.raw_native_projection import project


class RawNativeProjectionTest(unittest.TestCase):
    def test_active_command_and_lsq_fit_are_distinct_but_coupled(self):
        q = torch.tensor([[.1, .2, .3, 0, 0, 0, .2, .8]], dtype=torch.float32)
        spec = {"active": ["a"], "coupling": {"b": {"driver": "a", "multiplier": 2., "offset": 0.}}}
        raw, native = project(q, ["a", "b"], [0., 0.], [1., 2.], spec)
        self.assertTrue(torch.equal(raw[:, :6], q[:, :6]))
        self.assertTrue(torch.equal(native[:, :6], q[:, :6]))
        np.testing.assert_allclose(raw[0, 6:].numpy(), [.2, .4], atol=1e-6)
        np.testing.assert_allclose(native[0, 6:].numpy(), [.36, .72], atol=1e-6)

    def test_joint_limits_applied_to_active_and_passive(self):
        q = torch.tensor([[0., 0., 0., 0., 0., 0., 2., -2.]], dtype=torch.float32)
        spec = {"active": ["a"], "coupling": {"b": {"driver": "a", "multiplier": 2., "offset": .1}}}
        raw, native = project(q, ["a", "b"], [0., 0.], [1., 1.], spec)
        self.assertTrue(torch.isfinite(raw).all() and torch.isfinite(native).all())
        self.assertTrue(torch.all(raw[:, 6:] <= 1.) and torch.all(native[:, 6:] >= 0.))


if __name__ == "__main__":
    unittest.main()
