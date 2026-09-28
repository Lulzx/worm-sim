import unittest
from audit_dunn_timing_metadata import audit_frames


class TimingTests(unittest.TestCase):
    def test_coverage_and_overlap(self):
        r=audit_frames(list(range(100)),2,[{'stim_on':4,'stim_off':6},{'stim_on':20,'stim_off':24},{'stim_on':90,'stim_off':92}])
        self.assertFalse(r['events'][0]['history_available'])
        self.assertFalse(r['events'][2]['forecast_available'])
        self.assertEqual(r['events'][0]['other_stimuli_overlapping_window'],[1])
        self.assertEqual(r['events_with_full_frame_clock_window'],1)
        self.assertEqual(r['events'][1]['duration_seconds_frame_clock'],4.)

    def test_invalid_clocks_and_events(self):
        for frames,z,events in [([0,1,1,3],2,[]),([0,1,2],2,[]),([0,float('nan')],1,[]),
                                ([0,1,2,3],2,[{'stim_on':1,'stim_off':3}]),
                                ([0,1,2,3],2,[{'stim_on':0,'stim_off':4}])]:
            with self.assertRaises(ValueError):audit_frames(frames,z,events)


if __name__=='__main__':unittest.main()
