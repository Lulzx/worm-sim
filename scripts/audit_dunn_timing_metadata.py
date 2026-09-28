#!/usr/bin/env python3
"""Check acquisition timestamps and delivered-event coverage without response decoding."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from inspect_dunn_pickle_structure import load_structure


def audit_frames(frames, zsize, events, history=10., forecast=20.):
    if type(zsize) is not int or zsize <= 0:
        raise ValueError('positive integer zsize required')
    t = np.asarray(frames, dtype=float)
    if t.ndim != 1 or len(t) < 2 or len(t) % zsize or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('finite strictly increasing complete frame timestamps required')
    if not np.isfinite([history, forecast]).all() or min(history, forecast) <= 0:
        raise ValueError('positive windows required')
    intervals = []
    for event in events:
        on, off = event['stim_on'], event['stim_off']
        if type(on) is not int or type(off) is not int or not 0 <= on < off < len(t):
            raise ValueError('event frame bounds invalid')
        if on % zsize:
            raise ValueError('onset not on volume boundary')
        intervals.append((float(t[on]), float(t[off])))
    spacing = np.diff(t)
    gap_indices = np.flatnonzero(spacing > 2*np.median(spacing))
    gaps = [{'preceding_frame':int(k), 'following_frame':int(k+1),
             'start_seconds':float(t[k]), 'end_seconds':float(t[k+1]),
             'duration_seconds':float(spacing[k])} for k in gap_indices]
    rows = []
    for i, (on, off) in enumerate(intervals):
        left, right = on-history, on+forecast
        others = [j for j,(a,b) in enumerate(intervals) if j != i and a < right and b > left]
        rows.append({'event_index':i, 'onset_frame':events[i]['stim_on'],
                     'offset_frame':events[i]['stim_off'], 'onset_seconds_frame_clock':on,
                     'duration_seconds_frame_clock':off-on,
                     'history_available':bool(left >= t[0]),
                     'forecast_available':bool(right <= t[-1]),
                     'other_stimuli_overlapping_window':others,
                     'frame_gaps_overlapping_window':[j for j,g in enumerate(gaps)
                         if g['start_seconds'] < right and g['end_seconds'] > left]})
    spacing = np.diff(t); volumes = t[::zsize]; volume_spacing=np.diff(volumes)
    return {'zsize':zsize, 'frame_count':len(t), 'volume_count':len(volumes), 'history_seconds':history,
            'forecast_seconds':forecast, 'frame_spacing_min':float(spacing.min()),
            'frame_spacing_median':float(np.median(spacing)), 'frame_spacing_max':float(spacing.max()),
            'volume_start_spacing_min':float(volume_spacing.min()) if len(volume_spacing) else None,
            'volume_start_spacing_median':float(np.median(volume_spacing)) if len(volume_spacing) else None,
            'volume_start_spacing_max':float(volume_spacing.max()) if len(volume_spacing) else None,
            'frame_gaps_over_twice_median':len(gaps), 'frame_gaps':gaps,
            'events':rows, 'events_with_full_frame_clock_window':sum(r['history_available'] and r['forecast_available'] for r in rows),
            'events_without_other_stimuli_in_window':sum(not r['other_stimuli_overlapping_window'] for r in rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['pickle','download-receipt','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise ValueError('output exists')
    raw=Path(a.pickle).read_bytes();receipt_raw=Path(a.download_receipt).read_bytes();receipt=json.loads(receipt_raw)
    sha=hashlib.sha256(raw).hexdigest()
    if len(raw)!=receipt['bytes'] or sha!=receipt['sha256']:raise ValueError('download hash differs')
    root,_=load_structure(raw);md=root.state['md']
    result=audit_frames(md['frame_time_list'],md['gooey_args']['zsize'],md['stim_metadata']['stim_param_list'])
    result.update(schema_version=2,pickle_sha256=sha,download_receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
                  source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['audit_dunn_timing_metadata.py','inspect_dunn_pickle_structure.py']},
                  convention='Delivered stim_on/stim_off interpreted as zero-based raw-frame indices, with offset as exclusive boundary; 10 s before onset and 20 s after onset, half-open stimulus overlap intervals.',
                  scope='Metadata-only frame-clock screen. No response/timevec payload decoded. Does not validate synchronization to processed fluorescence, optical timing, missing neurons, independent animals or cohort eligibility.')
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))


if __name__=='__main__':main()
