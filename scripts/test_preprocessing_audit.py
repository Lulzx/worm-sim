#!/usr/bin/env python3
"""Checks for the preprocessing dependency audit, using small synthetic arrays."""
import unittest
import numpy as np
from audit_wormwideweb_preprocessing import audit_arrays

class AuditTests(unittest.TestCase):
    def test_detects_global_scaling_and_future_dependency(self):
        original = np.array([[float(t), float(t*t+2)] for t in range(30)])
        z = (original-original.mean(axis=0))/original.std(axis=0,ddof=1)
        result = audit_arrays(original,z,np.arange(30))
        self.assertTrue(result['matches_whole_recording_zscore_at_1e_minus_10'])
        change = result['future_only_counterfactual']
        self.assertEqual(change['original_prefix_max_change'],0.0)
        self.assertGreater(change['normalized_prefix_max_change'],0.1)
        prefix = (original-original[:11].mean(axis=0))/original[:11].std(axis=0,ddof=1)
        self.assertFalse(audit_arrays(original,prefix,np.arange(30))['matches_whole_recording_zscore_at_1e_minus_10'])

    def test_invalid_shapes_values_and_time_fail(self):
        x=np.arange(20.0).reshape(10,2)
        for original,z,t in [(x,x[:-1],np.arange(10)),(x,x,np.zeros(10)),(x*0,x,np.arange(10)),(x*np.nan,x,np.arange(10))]:
            with self.assertRaises(ValueError):audit_arrays(original,z,t)

if __name__=='__main__':unittest.main()
