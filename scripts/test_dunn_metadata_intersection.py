import copy
import unittest
from combine_dunn_metadata_screens import combine


class IntersectionTests(unittest.TestCase):
    def fixture(self):
        s={'pickle_sha256':'abc','events':[dict(event_index=0,onset_volume_index=2,
            inside_named_expressing_labels=['SMDVL'],inside_missing_identity_count=0)]}
        t={'pickle_sha256':'abc','zsize':12,'events':[dict(event_index=0,onset_frame=24,
            history_available=True,forecast_available=True,
            other_stimuli_overlapping_window=[],frame_gaps_overlapping_window=[])]}
        return s,t

    def test_all_conditions_required(self):
        s,t=self.fixture()
        self.assertEqual(combine(s,t)['counts']['joint_screen'],1)
        for key,value in [('history_available',False),('forecast_available',False),
                          ('other_stimuli_overlapping_window',[1]),('frame_gaps_overlapping_window',[0])]:
            changed=copy.deepcopy(t);changed['events'][0][key]=value
            self.assertEqual(combine(s,changed)['counts']['joint_screen'],0)
        s['events'][0]['inside_missing_identity_count']=1
        self.assertEqual(combine(s,t)['counts']['joint_screen'],0)

    def test_mismatched_lineage(self):
        for change in ['hash','onset','index','duplicate']:
            s,t=self.fixture()
            if change=='hash':t['pickle_sha256']='different'
            if change=='onset':t['events'][0]['onset_frame']=25
            if change=='index':t['events'][0]['event_index']=1
            if change=='duplicate':s['events'].append(s['events'][0])
            with self.assertRaises(ValueError):combine(s,t)


if __name__=='__main__':unittest.main()
