from __future__ import annotations



from driveaudit_eval.trajectory.schema import share_fixed_metric_frame, validate_unique_agent_ids


def select_protocol(generated, ground_truth):
    reasons = []
    if not ground_truth:
        return "A", ["paired ground truth is absent"]
    gen = {track.agent_id: track for track in generated if track.metadata.get("predict", True)}
    gt = {track.agent_id: track for track in ground_truth if track.metadata.get("predict", True)}
    if set(gen) != set(gt):
        reasons.append("agent_id sets do not match exactly")
    for agent_id in sorted(set(gen) & set(gt)):
        a, b = gen[agent_id], gt[agent_id]
        if a.coordinate_system != b.coordinate_system:
            reasons.append("coordinate_system mismatch for %s" % agent_id)
        if not share_fixed_metric_frame(a, b):
            reasons.append("shared fixed metric coordinate frame is unavailable for %s" % agent_id)
        overlap_start = max(a.timestamps[a.valid_mask].min(initial=float("inf")), b.timestamps[b.valid_mask].min(initial=float("inf")))
        overlap_end = min(a.timestamps[a.valid_mask].max(initial=-float("inf")), b.timestamps[b.valid_mask].max(initial=-float("inf")))
        # A shared single observation preserves pairing identity. The paired
        # metric kernel reports this actor as unavailable when it lacks enough
        # points; it must not disable supported actors and scene distributions.
        if not overlap_start <= overlap_end:
            reasons.append("no valid time overlap for %s" % agent_id)
    return ("A", reasons) if reasons else ("B", ["strict ids, temporal support, and coordinates are compatible"])
