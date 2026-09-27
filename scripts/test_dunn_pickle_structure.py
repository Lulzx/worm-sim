import os
import pickle
import tempfile
import unittest
from pathlib import Path
from inspect_dunn_pickle_structure import inspect


class Recording:
    pass


class SideEffect:
    def __reduce__(self):
        return os.system, ('touch should-not-exist',)


class StructureTests(unittest.TestCase):
    def test_numpy_arrays_remain_opaque(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('optional NumPy fixture requires the JAX environment')
        record = Recording()
        record.md = {'alg_metadata': {'stim_param_list': []}}
        record.response = np.arange(12.0).reshape(3, 4)
        for protocol in (4, 5):
            with self.subTest(protocol=protocol):
                result = inspect(pickle.dumps(record, protocol=protocol))
                self.assertTrue(result['object_fields']['response'].startswith('opaque:'))
                self.assertNotIn('response', result['metadata_fields'])

    def test_globals_are_inert_and_values_not_exported(self):
        record = Recording()
        record.md = {'subject': 'private-value', 'alg_metadata': {'stim_param_list': []}}
        record.md['stim_metadata'] = {'stim_param_list': [
            {'stim_on': 1234, 'event': {'event_type': 'pulse', 'amplitude': 6789}},
            {'stim_on': 1235, 'event': {'event_type': 'pulse', 'amplitude': 6789}},
        ]}
        record.array = SideEffect()
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            try:
                os.chdir(tmp)
                result = inspect(pickle.dumps(record, protocol=4))
                self.assertFalse(Path('should-not-exist').exists())
            finally:
                os.chdir(previous)
        self.assertEqual(result['metadata_fields']['subject'], 'str')
        self.assertEqual(result['nested_metadata_fields']['alg_metadata'], {'stim_param_list': 'list'})
        self.assertNotIn('private-value', str(result))
        self.assertEqual(result['delivered_event_schema']['count'], 2)
        self.assertEqual(result['stimulus_list_lengths']['md.stim_metadata.stim_param_list'], 2)
        self.assertEqual(result['stimulus_list_lengths']['md.alg_metadata.stim_param_list'], 0)
        self.assertEqual(result['delivered_event_schema']['event_type_counts'], {'pulse': 2})
        self.assertEqual(result['delivered_event_schema']['field_types']['event.amplitude'], ['int'])
        self.assertNotIn('6789', str(result))
        self.assertTrue(result['object_fields']['array'].startswith('opaque:'))

    def test_extension_and_wrong_root_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unsupported reference'):
            inspect(b'\x80\x02\x82\x01.')
        with self.assertRaisesRegex(ValueError, 'dictionary state'):
            inspect(pickle.dumps({'md': {}}))


if __name__ == '__main__':
    unittest.main()
