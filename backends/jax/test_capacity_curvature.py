import argparse
import unittest
import numpy as np
from diagnose_capacity_curvature import directional_summary, positive_step


class CurvatureTests(unittest.TestCase):
    def test_quadratic(self):
        matrix = np.diag([2., 8.])
        x = np.array([1., -2.]); d = np.array([0.6, 0.8])
        def evaluate(v):
            return float(v @ matrix @ v / 2), matrix @ v
        value, gradient = evaluate(x)
        for h in [0.001, 0.0005]:
            result = directional_summary(value, gradient, d, evaluate(x+h*d), evaluate(x-h*d), h)
            self.assertAlmostEqual(result['central_value_slope'], gradient @ d, places=9)
            self.assertAlmostEqual(result['gradient_secant_curvature'], d @ matrix @ d, places=9)
            self.assertAlmostEqual(result['central_value_curvature'], d @ matrix @ d, places=6)

    def test_step_argument(self):
        self.assertEqual(positive_step("1e-6"), 1e-6)
        for value in ["nan", "inf", "0", "-1"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                positive_step(value)

    def test_invalid_probes(self):
        for step, direction in [(0., np.array([1.])), (0.1, np.array([2.]))]:
            with self.assertRaises(ValueError):
                directional_summary(1., np.array([1.]), direction, (1., np.array([1.])), (1., np.array([1.])), step)
        with self.assertRaises(ValueError):
            directional_summary(1., np.array([1.]), np.array([1.]), (float('nan'), np.array([1.])), (1., np.array([1.])), .01)


if __name__ == '__main__':
    unittest.main()
