"""One scalar FPS condition for the retained, potentially irregular, clip.

No frames or trajectories are resampled. A scalar embedding cannot express
irregular exposures; the clip mean preserves duration and the evidence records
the complete native times. Non-12 values differ from the checkpoint's video
training rate and require an actual generation smoke check.
"""
import json
from pathlib import Path


def conditioning_timing(scene):
    path=Path(scene)/'timeline.json'
    timeline=json.loads(path.read_text())
    if timeline.get('unit')!='microseconds_relative_to_frame_0':
        raise ValueError('Unsupported native timeline unit: '+str(path))
    frames=timeline['frames']
    indices=[r['frame_index'] for r in frames]
    if indices!=list(range(90)):
        raise ValueError('Expected the original 90 ordered timeline frames: '+str(path))
    times=[r['reference_time_us'] for r in frames]
    if any(type(t) is not int for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Native reference times must be strictly increasing integer microseconds')
    used=times[1:]
    intervals=[b-a for a,b in zip(used,used[1:])]
    nominal_fps=timeline.get('nominal_contract_fps', timeline.get('fps'))
    if nominal_fps is not None and nominal_fps != 12:
        raise ValueError('Expected nominal contract FPS 12: '+str(path))
    return dict(fps=(len(used)-1)*1e6/(used[-1]-used[0]),
                nominal_contract_fps=nominal_fps,
                fps_conditioning_policy='retained_clip_mean_fps_no_resampling',
                input_frame_indices=list(range(1,90)),source_reference_times_us=used,
                nonuniform_native_intervals=max(intervals)-min(intervals)>2,
                minimum_native_interval_us=min(intervals),maximum_native_interval_us=max(intervals),
                resampled=False,checkpoint_video_training_fps=12)
