import copy
import unittest
import numpy as np
from test_planted_truth import fixture
from planted_truth import noise_ceiling
from trial_split import aggregate, check_mirror, split, expectation


def export(m, t, data):
    t = copy.deepcopy(t)
    names = t['names']; trials = {x['id']: x for x in data['trials']}
    g = t['groups'][0]
    t['groups'] = [aggregate([trials[i] for i in g['training_trials']], g['target'], names, g['recording']['times'])]
    return t


def data4():
    trace = lambda n, v, w: {'neuron': n, 'values': v, 'provenance': {'id_confidence': w}}
    times = [0., .05, .1]
    rng = np.random.default_rng(1)
    return {'trials': [{'id': f't{k}', 'stimulated_neuron': 'A', 'recording': {'times': times, 'traces': [
        trace('A', [0.]+list(rng.normal(size=2)*.01), 1.), trace('B', [0.]+list(rng.normal(size=2)*.01), .5 if k % 2 else 1.)]}}
        for k in range(4)]}


class TrialSplitTests(unittest.TestCase):
    def test_aggregate_matches_weighted_moments(self):
        m, g, t, data = fixture()
        for x in data['trials']:
            x['recording']['times'] = [0., .05, .1]
        a = aggregate(data['trials'], 0, t['names'], [0., .05, .1])
        np.testing.assert_allclose(a['recording']['traces'][0]['values'], [0., .01, .03], atol=1e-15)
        np.testing.assert_allclose(a['recording']['traces'][1]['values'], [0., .02, .02], atol=1e-15)
        self.assertEqual([x['provenance']['id_confidence'] for x in a['recording']['traces']], [1., .5])
        self.assertAlmostEqual(a['sample_weight'], 9.)
        # m2: A has 2*(.01^2)*2 samples, B has .5*2*(.01^2)*2 samples.
        self.assertAlmostEqual(a['irreducible_mse'], (4e-4+2e-4)/9., places=15)
        with self.assertRaises(ValueError):
            aggregate(data['trials'], 1, t['names'], [0., .05, .1])

    def test_split_is_disjoint_complete_and_mirror_checked(self):
        m, g, t, _ = fixture(); data = data4()
        t['groups'][0]['training_trials'] = [f't{k}' for k in range(4)]
        t = export(m, t, data)
        check_mirror(t, data, ['A'])
        fit, held = split(t, data, ['A'], 0)
        self.assertEqual(fit['groups'][0]['training_trials'], ['t0', 't2'])
        self.assertEqual(held['groups'][0]['training_trials'], ['t1', 't3'])
        self.assertEqual(fit['trial_split']['half'], 'fit')
        bad = copy.deepcopy(t); bad['groups'][0]['recording']['traces'][0]['values'][1] += 1e-6
        with self.assertRaises(ValueError):
            check_mirror(bad, data, ['A'])

    def test_expectation_matches_noise_ceiling(self):
        m, g, t, _ = fixture(); data = data4()
        t['groups'][0]['training_trials'] = [f't{k}' for k in range(4)]
        t = export(m, t, data)
        trials = {x['id']: x for x in data['trials']}
        self.assertAlmostEqual(expectation(t['groups'], trials), noise_ceiling(t, data, ['A'])['expected_truth_capture'], places=12)


if __name__ == '__main__':
    unittest.main()
