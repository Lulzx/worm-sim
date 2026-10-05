from collections import Counter
import unittest
import numpy as np
from mirror_reliability import Model, mirror_map, mirror_pairs, rewire
from probe_molecular_sign_agreement import group_table, permutation_test


class MirrorTests(unittest.TestCase):
    def test_suffix_swap_requires_existing_partner(self):
        m = mirror_map({'AVAL', 'AVAR', 'AVL', 'PVR', 'RMDDL', 'RMDDR', 'DA1'})
        self.assertEqual(m['AVAL'], 'AVAR')
        self.assertEqual(m['RMDDR'], 'RMDDL')
        for single in ['AVL', 'PVR', 'DA1']:
            self.assertEqual(m[single], single)

    def test_pairs_exclude_self_mirrors_and_keep_absent_sides(self):
        m = mirror_map({'AL', 'AR', 'B'})
        pairs = mirror_pairs({('AL', 'B'): 2., ('B', 'B'): 1.}, m, True)
        self.assertEqual(pairs, [(('AL', 'B'), ('AR', 'B'), 2., 0.)])
        gaps = mirror_pairs({('AL', 'AR'): 3.}, m, False)
        self.assertEqual(gaps, [])

    def test_rewiring_preserves_degrees(self):
        rng = np.random.default_rng(0)
        edges = [(int(a), int(b)) for a, b in rng.integers(30, size=(200, 2)) if a != b]
        edges = list(dict.fromkeys(edges))
        out = rewire(edges, True, rng)
        self.assertEqual(Counter(a for a, _ in edges), Counter(a for a, _ in out))
        self.assertEqual(Counter(b for _, b in edges), Counter(b for _, b in out))
        self.assertEqual(len(set(out)), len(out))
        self.assertNotEqual(set(edges), set(out))

    def test_model_recovers_simulated_parameters(self):
        truth = np.array([0., np.log(1.2), np.log(.3), -.5, -1.])
        sim = Model([1.], [1.]).simulate(truth, 6000, np.random.default_rng(1))
        model = Model(sim[:, 0], sim[:, 1])
        from scipy.optimize import minimize
        fit = minimize(model.negloglik, np.array([.5, 0., -1., 0., 0.]), method='Nelder-Mead',
                       options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 8000})
        self.assertTrue(fit.success)
        self.assertLess(model.negloglik(fit.x), model.negloglik(truth))
        np.testing.assert_allclose(fit.x[:2], truth[:2], atol=.2)
        np.testing.assert_allclose(np.exp(fit.x[1:3]), np.exp(truth[1:3]), atol=.25)


class ProbeTests(unittest.TestCase):
    def test_tied_groups_and_mixed_labels(self):
        evidence = [{'state': 'inhibitory', 'transmitters': ['GABA']},
                    {'state': 'conflicting', 'transmitters': ['GABA']},
                    {'state': 'excitatory', 'transmitters': ['ACh']},
                    {'state': 'inhibitory', 'transmitters': ['ACh']}]
        keys, labels, strata, mixed = group_table(np.array([5, 5, 7, 7]), evidence)
        self.assertEqual(keys, [5, 7])
        self.assertEqual(labels, [-1, None])
        self.assertEqual(strata, ['GABA', 'ACh'])
        self.assertEqual(mixed, 1)

    def test_detects_planted_agreement_but_not_stratified_shrinkage(self):
        rng = np.random.default_rng(2)
        n = 600
        labels = rng.choice([-1., 0., 0., 1.], n)
        start = rng.choice([-.5, .5], n)
        strata = np.where(start > 0, 'pos', 'neg')
        # Shrinkage only: movement is opposite to the start, unrelated to labels,
        # but labels are deliberately enriched among positive starts.
        labels[start > 0] = np.where(rng.random((start > 0).sum()) < .5, -1., labels[start > 0])
        shrink = -.1*start
        p_shrink = permutation_test(labels, shrink, strata, 2000, rng)[3]
        self.assertGreater(p_shrink, .05)
        planted = np.where(labels != 0, labels, rng.choice([-1., 1.], n))*rng.uniform(.01, .2, n)
        p_planted = permutation_test(labels, planted, strata, 2000, rng)[3]
        self.assertLess(p_planted, .001)


if __name__ == '__main__':
    unittest.main()
