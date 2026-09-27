import unittest
import numpy as np
from audit_capacity_stationarity import summarize


class StationarityTests(unittest.TestCase):
    def test_frozen_coordinates_excluded_and_native_families_grouped(self):
        gradient = {'groups': np.array([3., 900., 4.]), 'observation': {'log_gain': np.array([12., 500.])}}
        active = {'groups': np.array([True, False, True]), 'observation': {'log_gain': np.array([True, False])}}
        r = summarize(gradient, active, ['rest/a', 'rest/b', 'tau/a'])
        self.assertEqual(r['active_coordinates'], 3)
        self.assertEqual(r['active_gradient_l2'], 13.)
        self.assertEqual(r['active_gradient_linf'], 12.)
        self.assertEqual([x['family'] for x in r['families']], ['native/rest', 'native/tau', 'observation/log_gain'])

    def test_bad_masks_names_and_nonfinite_reject(self):
        for gradient, active, names in [
            ({'groups': np.array([1.])}, {'groups': np.array([1])}, ['rest/a']),
            ({'groups': np.array([1.])}, {'groups': np.array([True])}, []),
            ({'groups': np.array([np.nan])}, {'groups': np.array([True])}, ['rest/a']),
        ]:
            with self.assertRaises(ValueError):
                summarize(gradient, active, names)


if __name__ == '__main__':
    unittest.main()
