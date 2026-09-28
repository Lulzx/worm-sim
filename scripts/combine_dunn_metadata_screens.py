#!/usr/bin/env python3
"""Intersect spatial and timing metadata screens, without declaring cohort eligibility."""
import argparse
import hashlib
import json
from pathlib import Path


def combine(spatial, timing):
    if spatial['pickle_sha256'] != timing['pickle_sha256']:
        raise ValueError('pickle hashes differ')
    sr, tr = spatial['events'], timing['events']
    if ([r['event_index'] for r in sr] != list(range(len(sr))) or
        [r['event_index'] for r in tr] != list(range(len(sr)))):
        raise ValueError('event indices differ or are not unique and ordered')
    rows=[]
    for s,t in zip(sr,tr):
        if t['onset_frame'] != s['onset_volume_index'] * timing['zsize']:
            raise ValueError('onset indices differ')
        geometry = len(s['inside_named_expressing_labels']) == 1 and s['inside_missing_identity_count'] == 0
        coverage = t['history_available'] and t['forecast_available']
        isolated = not t['other_stimuli_overlapping_window']
        gap_free = not t['frame_gaps_overlapping_window']
        rows.append(dict(event_index=s['event_index'], spatial_screen=geometry,
                         full_window=coverage, no_other_stimulus=isolated,
                         no_detected_frame_gap=gap_free,
                         joint_screen=geometry and coverage and isolated and gap_free))
    return dict(pickle_sha256=timing['pickle_sha256'], events=rows,
                counts={k:sum(r[k] for r in rows) for k in
                        ['spatial_screen','full_window','no_other_stimulus','no_detected_frame_gap','joint_screen']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['spatial','timing','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args()
    raw={k:Path(getattr(a,k)).read_bytes() for k in ['spatial','timing']}
    result=combine(*(json.loads(raw[k]) for k in ['spatial','timing']))
    result.update(schema_version=1, input_sha256={k:hashlib.sha256(v).hexdigest() for k,v in raw.items()},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Metadata intersection only; does not establish independent animals, single-neuron stimulation, response synchronization or test-cohort eligibility.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result['counts']))


if __name__=='__main__':main()
