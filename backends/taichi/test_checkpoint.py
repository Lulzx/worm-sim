#!/usr/bin/env python3
"""Boundary ownership, full adjoint carry, and repeated-call regression tests."""
import json
import sys
from pathlib import Path
import unittest
import numpy as np
import taichi as ti
from checkpoint import CheckpointLevel0
from level0 import Level0

fixture_path = sys.argv.pop(1)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, default_fp=ti.f64, fast_math=False, debug=True,
                cpu_max_num_threads=1, ad_stack_size=256, offline_cache=False)
        cls.fixture = json.loads(Path(fixture_path).read_text())

    def test_boundaries_and_repeated_calls(self):
        f = self.fixture
        full = Level0(f, batch=2, dtype=ti.f64)
        expected_loss, expected_gradient = full.value_and_grad(validation=True)
        np.testing.assert_allclose(expected_gradient, f['reference']['gradient'], atol=1e-11, rtol=1e-8)
        for interval in [1, 7, f['steps'], f['steps']+1]:
            with self.subTest(interval=interval):
                model = CheckpointLevel0(f, interval, batch=2, dtype=ti.f64)
                for _ in range(2):
                    loss, gradient = model.value_and_grad(validation=True)
                    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-11, rtol=1e-8)
                    self.assertAlmostEqual(loss, expected_loss, places=12)
                # A second parameter vector must regenerate checkpoints and
                # clear the accumulated parameter/boundary derivatives.
                changed = np.array(f['parameters_raw'])
                changed[0] += 0.15
                changed[-1] -= 0.2
                full.raw.from_numpy(changed)
                model.raw.from_numpy(changed)
                changed_loss, changed_gradient = full.value_and_grad(validation=True)
                loss, gradient = model.value_and_grad(validation=True)
                np.testing.assert_allclose(gradient, changed_gradient, atol=1e-11, rtol=1e-8)
                self.assertAlmostEqual(loss, changed_loss, places=12)
                full.raw.from_numpy(np.asarray(f['parameters_raw']))
                # A primal-only pass after validation must not inherit stale
                # validator kernel modes from repeatedly invoked advance().
                self.assertAlmostEqual(model.forward(), changed_loss, places=12)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            CheckpointLevel0(self.fixture, 0)


if __name__ == '__main__':
    unittest.main()
