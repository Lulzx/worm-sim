import unittest
import numpy as np
from audit_dunn_spatial_metadata import audit, coordinates, spot_members
from inspect_dunn_pickle_structure import load_structure
from test_dunn_pickle_structure import Recording
import pickle


class SpatialTests(unittest.TestCase):
    def test_registration_and_unlabeled_soma_are_preserved(self):
        r = Recording()
        r.quant_method = 'gcamp-extractor'
        r.ID1 = np.array(['SMDDL', float('nan'), 'AVAL'], dtype=object)
        r.x = np.array([[0., 4., 10.], [0., 4., 10.]])
        r.y = np.zeros((2, 3))
        raw = {'stim_on': 2, 'stim_off': 4,
               'event': {'event_type': 'circle-button', 'x': 2, 'y': 1, 'stim_diameter': 10}}
        r.stim_param_list = [{'stim_on': 2, 'stim_off': 4,
                             'event': {'event_type': 'circle-button', 'x': 0, 'y': 0, 'stim_diameter': 10}}]
        r.md = {'gooey_args': {'subject_strain': 'FC121', 'zsize': 2},
                'stim_metadata': {'stim_param_list': [raw]},
                'postprocessing': {'moco': {'registration_method': 'manual_rigid_xy',
                                           'registration_global_offset': {'x': [2, 2], 'y': [1, 1]}}}}
        root, _ = load_structure(pickle.dumps(r, protocol=4))
        result = audit(root.state)
        self.assertEqual(result['events'][0]['inside_named_expressing_labels'], ['SMDDL'])
        self.assertEqual(result['events_with_unlabeled_somata_inside'], 1)
        root.state['stim_param_list'][0]['event']['x'] = 1
        with self.assertRaisesRegex(ValueError, 'registration correction'):
            audit(root.state)

    def test_coordinates_and_response_rejection(self):
        r = Recording()
        r.md = {}
        for dtype in ('<f8', '>f8'):
            for order in ('C', 'F'):
                r.x = np.array([[1, 2], [3, 4]], dtype=dtype, order=order)
                r.dff = np.ones((2, 2))
                root, _ = load_structure(pickle.dumps(r, protocol=4))
                np.testing.assert_array_equal(coordinates(root.state, 'x'), r.x)
                with self.assertRaisesRegex(ValueError, 'only x and y'):
                    coordinates(root.state, 'dff')
        a = root.state['x']
        a.state = (*a.state[:4], a.state[4][:-1])
        with self.assertRaises(ValueError):
            coordinates(root.state, 'x')

    def test_circle_boundary(self):
        self.assertEqual(spot_members(np.array([0., 3., 3.01]), np.array([0., 4., 4.]), 0., 0., 10.), [0, 1])
        with self.assertRaises(ValueError):
            spot_members(np.array([0.]), np.array([0.]), 0., 0., -1.)


if __name__ == '__main__':
    unittest.main()
