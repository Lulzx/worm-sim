import unittest
import copy
from audit_lbfgs_prefix import complete_rows, compare_value, check


class PrefixTests(unittest.TestCase):
    def test_only_declared_budgets_may_change(self):
        reference={k:{} for k in ['input_sha256','targets','training_trials','warm_start','configuration','fit_config','bounds','backend_source_sha256','jax','scipy']}
        reference.update(input_sha256={'warm_start':'parent'},targets=['A'],fit_config={'dt':.005},
            fitting_optimizer={'max_evaluations':201,'max_iterations':200,'maxls':20,'maxcor':20,'ftol':1e-12,'gtol':1e-9})
        candidate=copy.deepcopy(reference);candidate['fitting_optimizer'].update(max_evaluations=1001,max_iterations=1000)
        declaration=dict(candidate['fitting_optimizer'],parent_sha256='parent',dt=.005,targets=['A'])
        check(reference,candidate,declaration)
        for mutate in [lambda c:c['fit_config'].update(dt=.0025),
                       lambda c:c['input_sha256'].update(warm_start='other'),
                       lambda c:c['backend_source_sha256'].update(source='changed'),
                       lambda c:c['fitting_optimizer'].update(max_evaluations=1002),
                       lambda c:c['fitting_optimizer'].update(gtol=1e-5)]:
            invalid=copy.deepcopy(candidate);mutate(invalid)
            with self.assertRaises(ValueError):check(reference,invalid,declaration)

    def test_partial_line_and_sequence(self):
        self.assertEqual(complete_rows(b'{"evaluation":1}\n{"evalu', 'evaluation'),[{'evaluation':1}])
        with self.assertRaises(ValueError):complete_rows(b'{"evaluation":2}\n','evaluation')
        with self.assertRaises(ValueError):complete_rows(b'not json\n','evaluation')

    def test_strict_fields_finiteness_and_tolerance(self):
        compare_value({'mse':.03,'finite':True,'evaluation':2},{'mse':.03+1e-12,'finite':True,'evaluation':2})
        for bad in [{'mse':float('nan'),'finite':True,'evaluation':2},
                    {'mse':.0301,'finite':True,'evaluation':2},
                    {'mse':.03,'finite':False,'evaluation':2},
                    {'mse':.03,'finite':True,'evaluation':3},
                    {'mse':.03,'evaluation':2}]:
            with self.assertRaises(ValueError):compare_value({'mse':.03,'finite':True,'evaluation':2},bad)


if __name__=='__main__':unittest.main()
