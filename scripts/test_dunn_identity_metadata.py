import unittest
from inspect_dunn_identity_metadata import identity_labels
from inspect_dunn_pickle_structure import Opaque


def fixture(labels):
    array = Opaque()
    array.pickle_global = ('numpy.core.multiarray', '_reconstruct')
    dtype = Opaque('O8')
    dtype.pickle_global = ('numpy', 'dtype')
    array.state = (1, (len(labels),), dtype, False, labels)
    return array


class IdentityTests(unittest.TestCase):
    def test_names_missing_and_duplicates(self):
        total, missing, names = identity_labels(fixture(['AVAL', None, float('nan'), '', 'AVAL', 'unknown']))
        self.assertEqual((total, missing), (6, 3))
        self.assertEqual(names, {'AVAL': 2, 'unknown': 1})

    def test_unknown_layout_or_values_reject(self):
        with self.assertRaises(ValueError):
            identity_labels(fixture([12.5]))
        with self.assertRaises(ValueError):
            identity_labels(['AVAL'])
        a = fixture(['AVAL'])
        a.state = (1, (2,), a.state[2], False, ['AVAL'])
        with self.assertRaises(ValueError):
            identity_labels(a)


if __name__ == '__main__':
    unittest.main()
