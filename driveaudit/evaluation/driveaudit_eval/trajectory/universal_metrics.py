"""Map-independent physical support for opt-in universal TRS."""
import itertools
import numpy as np
from driveaudit_eval.trajectory.collision_metrics import evaluate_collisions
from driveaudit_eval.trajectory.schema import share_fixed_metric_frame
from driveaudit_eval.trajectory.kinematic_metrics import kinematic_series


def evaluate_universal(tracks, config, collision_result=None):
    """Penalize nonphysical relative jumps, not naturally dangerous low TTC.

    A jointly observed relative displacement is feasible when its speed is no
    greater than the sum of the two frozen per-type teleport speed limits.
    Collision realism uses the existing inset-core background collision kernel.
    """
    samples = violations = pair_count = 0
    limits = config['validity']['teleport_speed_max']
    for a, b in itertools.combinations(tracks, 2):
        if not share_fixed_metric_frame(a, b):
            continue
        start_a = int(a.metadata.get('history_frames') or 0)
        start_b = int(b.metadata.get('history_frames') or 0)
        ma = {round(float(t), 9): i for i,t in enumerate(a.timestamps) if i >= start_a and a.valid_mask[i]}
        mb = {round(float(t), 9): i for i,t in enumerate(b.timestamps) if i >= start_b and b.valid_mask[i]}
        common = sorted(set(ma) & set(mb))
        if len(common) < 2:
            continue
        ia, ib = np.asarray([ma[t] for t in common]), np.asarray([mb[t] for t in common])
        adjacent = (np.diff(ia) == 1) & (np.diff(ib) == 1)
        dt = np.diff(common)[adjacent]
        relative = a.positions[ia] - b.positions[ib]
        speed = np.linalg.norm(np.diff(relative, axis=0)[adjacent], axis=1)/dt
        bound = float(limits[a.agent_type]) + float(limits[b.agent_type])
        samples += len(speed); violations += int(np.sum(~np.isfinite(speed) | (speed > bound)))
        pair_count += bool(len(speed))
    collision = evaluate_collisions(tracks, config.get('collision')) if collision_result is None else collision_result
    timestamps = {round(float(t), 9) for track in tracks
                  for i,t in enumerate(track.timestamps)
                  if i >= int(track.metadata.get('history_frames') or 0) and track.valid_mask[i]}
    kinematic_count = 0
    for track in tracks:
        start = int(track.metadata.get('history_frames') or 0)
        series = kinematic_series(track)
        kinematic_count += sum(int(np.isfinite(series[name][start:]).sum()) for name in
                               ('longitudinal_acceleration','lateral_acceleration','longitudinal_jerk','yaw_rate'))
    return dict(protocol='universal_v1', kinematic_sample_count=kinematic_count,
        interaction_plausibility=1-violations/samples if samples else None,
        interaction_sample_count=samples, nonphysical_pair_transition_count=violations,
        evaluated_pair_count=pair_count, teleport_speed_limits_mps=dict(limits),
        collision_realism=1-float(collision['collision_frame_ratio']) if timestamps else None,
        collision_sample_count=len(timestamps), collision_frame_ratio=collision['collision_frame_ratio'],
        collision_core_inset_m=collision['collision_core_inset_m'],
        interaction_policy='relative displacement feasibility; low TTC is not itself a realism penalty',
        collision_policy='existing inset-core participant collision fraction over observed future timestamps')
