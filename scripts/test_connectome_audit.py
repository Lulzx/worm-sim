#!/usr/bin/env python3
"""Independent audit arithmetic checks, including missing and zero-confidence data."""
import unittest
import numpy as np
from audit_connectome_fit import impulse, trace_scores


class AuditTests(unittest.TestCase):
    def test_dense_impulse_routes_shared_input_after_initial_frame(self):
        model = {'dynamics': {'gaussian': {'dim': 2, 'transition': [0.5, 0.2, -0.1, 0.8]}, 'kernel': [1.0, 0.4]}}
        np.testing.assert_allclose(impulse(model, 0, 4), [[0., 0.], [1., 0.], [0.9, -0.1], [0.43, -0.17]], atol=1e-15)
        np.testing.assert_allclose(impulse(model, 1, 3), [[0., 0.], [0., 1.], [0.2, 1.2]], atol=1e-15)

    def test_scores_weight_samples_and_only_defined_correlations(self):
        trials = {'a': {'recording': {'times': [0., 1., 2.], 'traces': [
            {'neuron': 'X', 'values': [1., None, 3.], 'provenance': {'id_confidence': 0.5}},
            {'neuron': 'Y', 'values': [1., 2., 3.], 'provenance': {'id_confidence': 1.}},
            {'neuron': 'Z', 'values': [100., 100., 100.], 'provenance': {'id_confidence': 0.}},
        ]}}}
        prediction = {'trials': [{'id': 'a', 'times': [0., 1., 2.], 'fluorescence': {'X': [2., 999., 4.], 'Y': [0., 0., 0.], 'Z': [0., 0., 0.]}}]}
        score = trace_scores(trials, prediction)
        self.assertAlmostEqual(score['pooled_mse'], 15/4)
        self.assertAlmostEqual(score['zero_response_mse'], 19/4)
        self.assertAlmostEqual(score['macro_trace_correlation'], 1.)
        self.assertEqual(score['defined_trace_correlations'], 1)


if __name__ == '__main__':
    unittest.main()
