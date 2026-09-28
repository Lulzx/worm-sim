import unittest
import numpy as np
from audit_preparation_gradients import compare_vectors


class PreparationGradientTests(unittest.TestCase):
    def test_known_vectors_and_zero(self):
        r=compare_vectors([3.,4.],[6.,8.])
        self.assertEqual(r['difference_l2'],5.)
        self.assertEqual(r['relative_difference_l2'],1.)
        self.assertEqual(r['cosine'],1.)
        self.assertIsNone(compare_vectors([0.],[1.])['cosine'])
        self.assertIsNone(compare_vectors([0.],[1.])['relative_difference_l2'])

    def test_invalid_vectors(self):
        for a,b in [([],[]),([1.],[1.,2.]),([np.nan],[1.]),([[1.]],[[1.]])]:
            with self.assertRaises(ValueError):compare_vectors(a,b)


if __name__=='__main__':unittest.main()
