from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from driveaudit_eval.trajectory import NOT_AVAILABLE
from driveaudit_eval.trajectory.common import stats, wrap_angle


def _poly(records, s):
    if not records:
        return 0.0
    record = max((item for item in records if float(item.get("sOffset", item.get("s", 0))) <= s),
                 key=lambda item: float(item.get("sOffset", item.get("s", 0))), default=records[0])
    base = float(record.get("sOffset", record.get("s", 0)))
    ds = s - base
    return sum(float(record.get(key, 0)) * ds ** power for power, key in enumerate(("a", "b", "c", "d")))


def _geometry_point(geometry, ds):
    x, y, heading = (float(geometry.get(key)) for key in ("x", "y", "hdg"))
    arc = geometry.find("arc")
    if geometry.find("line") is None and arc is None:
        raise ValueError("unsupported OpenDRIVE geometry; only line and arc are evaluated")
    if arc is None or abs(float(arc.get("curvature", 0))) < 1e-12:
        return x + ds * math.cos(heading), y + ds * math.sin(heading), heading
    curvature = float(arc.get("curvature"))
    angle = heading + curvature * ds
    return (x + (math.sin(angle) - math.sin(heading)) / curvature,
            y - (math.cos(angle) - math.cos(heading)) / curvature, angle)


def parse_opendrive(path, sample_step=0.5):
    root = ET.parse(path).getroot()
    lane_samples = {}
    road_links = {}
    for road in root.findall("road"):
        road_id = road.get("id")
        link = road.find("link")
        road_links[road_id] = {
            tag: (link.find(tag).get("elementId") if link is not None and link.find(tag) is not None else None)
            for tag in ("predecessor", "successor")
        }
        sections = road.findall("./lanes/laneSection")
        offsets = [item.attrib for item in road.findall("./lanes/laneOffset")]
        geometries = road.findall("./planView/geometry")
        for geometry in geometries:
            g_start, length = float(geometry.get("s")), float(geometry.get("length"))
            samples = np.arange(0.0, length, sample_step).tolist() + [length]
            for ds in samples:
                s = g_start + ds
                section_idx = max((i for i, item in enumerate(sections) if float(item.get("s")) <= s), default=0)
                section = sections[section_idx]
                section_s = s - float(section.get("s"))
                x, y, heading = _geometry_point(geometry, ds)
                offset = _poly(offsets, s)
                for side_name, sign in (("left", 1), ("right", -1)):
                    cumulative = offset
                    lanes = sorted(section.findall("./%s/lane" % side_name), key=lambda lane: abs(int(lane.get("id"))))
                    for lane in lanes:
                        widths = [item.attrib for item in lane.findall("width")]
                        width = max(0.0, _poly(widths, section_s))
                        inner = cumulative
                        outer = cumulative + sign * width
                        center = (inner + outer) / 2.0
                        cumulative = outer
                        if lane.get("type") != "driving" or width <= 0:
                            continue
                        normal = np.array([-math.sin(heading), math.cos(heading)])
                        ref = np.array([x, y])
                        vector_lane = lane.find("./userData/vectorLane")
                        travel = vector_lane.get("travelDir") if vector_lane is not None else None
                        travel_heading = heading + (math.pi if int(lane.get("id")) > 0 else 0.0)
                        if travel == "backward":
                            travel_heading += math.pi
                        key = (road_id, section_idx, int(lane.get("id")))
                        lane_samples.setdefault(key, {"center": [], "left": [], "right": [], "heading": [], "road_id": road_id})
                        sample = lane_samples[key]
                        sample["center"].append((ref + center * normal).tolist())
                        boundaries = [ref + inner * normal, ref + outer * normal]
                        left_point, right_point = (boundaries if sign > 0 else boundaries[::-1])
                        sample["left"].append(left_point.tolist())
                        sample["right"].append(right_point.tolist())
                        sample["heading"].append(float(wrap_angle(travel_heading)))
    lanes = []
    polygons = []
    for key, sample in lane_samples.items():
        if len(sample["center"]) < 2:
            continue
        polygon = Polygon(sample["left"] + list(reversed(sample["right"]))).buffer(0)
        if polygon.is_empty:
            continue
        sample["key"] = "%s:%s:%s" % key
        sample["polygon"] = polygon
        sample["line"] = LineString(sample["center"])
        sample["center"] = np.asarray(sample["center"], dtype=float)
        sample["heading"] = np.asarray(sample["heading"], dtype=float)
        lanes.append(sample)
        polygons.append(polygon)
    return {
        "lanes": lanes,
        "drivable": unary_union(polygons) if polygons else Polygon(),
        "road_links": road_links,
        "signal_count": len(root.findall(".//signal")),
        "junction_count": len(root.findall("junction")),
        "semantic_support": {
            "driving_lanes": bool(lanes),
            "road_links": bool(road_links),
            "signals": bool(root.findall(".//signal")),
            "crosswalks": False,
            "reference_route": False,
            "signal_state": False,
        },
    }


def obb_polygon(position, heading, length, width):
    local = np.asarray([[length / 2, width / 2], [length / 2, -width / 2],
                        [-length / 2, -width / 2], [-length / 2, width / 2]])
    rotation = np.asarray([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
    return Polygon(local.dot(rotation.T) + np.asarray(position))


def _nearest_lane(position, map_data, heading=None):
    candidates = []
    for lane in map_data["lanes"]:
        distances = np.linalg.norm(lane["center"] - position, axis=1)
        idx = int(np.argmin(distances))
        candidates.append((float(distances[idx]), lane, idx))
    if not candidates:
        return None
    best_distance = min(value[0] for value in candidates)
    nearby = [value for value in candidates if value[0] <= best_distance + 1.5]
    if heading is None:
        return min(nearby, key=lambda value: value[0])
    # Intersections can contain overlapping connector centerlines. Use heading only to disambiguate spatially-near lanes.
    return min(nearby, key=lambda value: (abs(float(wrap_angle(heading - value[1]["heading"][value[2]]))), value[0]))


def evaluate_track_map_rules(track, map_data, minimum_occupancy=0.99, wrong_way_threshold=math.pi / 2):
    if track.coordinate_system != "world" and not (track.coordinate_system == "history_last_ego" and provider.coordinate_system == "history_last_ego"):
        return {"status": NOT_AVAILABLE, "reason": "track and map provider require a shared fixed coordinate frame"}
    if track.agent_type not in {"vehicle", "ego"}:
        return {"status": NOT_AVAILABLE, "reason": "vehicle drivable-lane rules are not applied to this agent type"}
    start = int(track.metadata.get("history_frames") or 0)
    indices = np.flatnonzero(track.valid_mask[start:]) + start
    if not len(indices) or map_data["drivable"].is_empty:
        return {"status": NOT_AVAILABLE, "reason": "no valid future points or no parsed driving-lane geometry"}
    occupancy, deviations, heading_errors, wrong_way, lane_ids, offroad, outside_distances = [], [], [], [], [], [], []
    for idx in indices:
        box = obb_polygon(track.positions[idx], track.headings[idx], track.shape["length"], track.shape["width"])
        ratio = float(box.intersection(map_data["drivable"]).area / box.area) if box.area else 0.0
        occupancy.append(ratio)
        offroad.append(ratio < minimum_occupancy)
        outside_distances.append(max((Point(corner).distance(map_data["drivable"]) for corner in box.exterior.coords[:-1]), default=0.0))
        nearest = _nearest_lane(track.positions[idx], map_data, track.headings[idx])
        if nearest:
            distance, lane, sample_idx = nearest
            error = float(abs(wrap_angle(track.headings[idx] - lane["heading"][sample_idx])))
            deviations.append(distance)
            heading_errors.append(error)
            wrong_way.append(error > wrong_way_threshold)
            lane_ids.append(lane["key"])
    return {
        "status": "available",
        "agent_type": track.agent_type,
        "minimum_obb_occupancy_threshold": float(minimum_occupancy),
        "drivable_area_occupancy_ratio": float(np.mean(occupancy)),
        "off_road_area_ratio": float(np.mean(1.0 - np.asarray(occupancy))),
        "off_road_frame_ratio": float(np.mean(offroad)),
        "off_road_agent": bool(any(offroad)),
        "off_road_indices": [int(indices[i]) for i, flag in enumerate(offroad) if flag],
        "evaluated_indices": indices.tolist(),
        "evaluated_timestamps": track.timestamps[indices].tolist(),
        "obb_occupancy_ratio_series": occupancy,
        "off_road_flags": offroad,
        "obb_outside_distance_series": outside_distances,
        "maximum_obb_outside_distance": float(max(outside_distances, default=0.0)),
        "offroad_at_map_entry": bool(offroad and offroad[0] and any(not flag for flag in offroad[1:])),
        "offroad_at_map_exit": bool(offroad and offroad[-1] and any(not flag for flag in offroad[:-1])),
        "lane_center_deviation": stats(deviations),
        "lane_heading_error": stats(heading_errors),
        "wrong_way_ratio": float(np.mean(wrong_way)) if wrong_way else NOT_AVAILABLE,
        "nearest_lane_ids": lane_ids,
        "route_compliance": NOT_AVAILABLE,
        "lane_connectivity_violation": NOT_AVAILABLE,
        "illegal_lane_change": NOT_AVAILABLE,
        "solid_line_crossing": NOT_AVAILABLE,
        "stop_line_compliance": NOT_AVAILABLE,
        "red_light_violation": NOT_AVAILABLE,
        "crosswalk_conflict": NOT_AVAILABLE,
        "expected_maneuver_compliance": NOT_AVAILABLE,
        "unavailable_reasons": {
            "route_compliance": "no submitted reference route",
            "lane_connectivity_violation": "nearest-lane labels alone do not establish a reliable lane transition",
            "illegal_lane_change": "lane-change intent and full boundary semantics are absent",
            "solid_line_crossing": "lane-boundary crossing reconstruction is not reliable from this map adapter",
            "stop_line_compliance": "stop-line semantics are absent",
            "red_light_violation": "time-aligned signal state is absent",
            "crosswalk_conflict": "crosswalk semantic polygons are absent",
            "expected_maneuver_compliance": "expected maneuver is absent",
        },
    }


def evaluate_track_map_provider(track, provider, minimum_occupancy=0.99,
                                wrong_way_threshold=math.pi / 2,
                                minimum_wrong_way_speed=0.5):
    """Evaluate one world-frame track through the shared MapProvider contract."""
    if track.coordinate_system != "world" and not (track.coordinate_system == "history_last_ego" and provider.coordinate_system == "history_last_ego"):
        return {"status": NOT_AVAILABLE, "reason": "track and map provider require a shared fixed coordinate frame"}
    start = int(track.metadata.get("history_frames") or 0)
    indices = np.flatnonzero(track.valid_mask[start:]) + start
    if not len(indices):
        return {"status": NOT_AVAILABLE, "reason": "no valid future points"}
    points = track.positions[indices]
    try:
        centers = provider.is_drivable(points)
        signed = provider.signed_distance_to_drivable(points)
        footprints = provider.footprint_coverage(points, track.headings[indices], track.shape["length"], track.shape["width"])
        road_boundary = provider.distance_to_road_boundary(points)
    except Exception as exc:
        return {"status": NOT_AVAILABLE, "reason": "map query failed: %s: %s" % (type(exc).__name__, exc)}
    footprint_offroad = footprints < float(minimum_occupancy)
    center_offroad = ~centers
    dt = float(track.dt or track.metadata.get("dt") or 0.0)
    lane_applies = track.agent_type in {"vehicle", "ego"}
    lane_available = provider.query_metadata().get("semantic_support", {}).get("directed_lane_centerlines") is not False
    lane_matches = provider.nearest_lane_center(points) if lane_applies else []
    matched = np.asarray([item.get("status") == "matched" for item in lane_matches], dtype=bool)
    lane_distances = np.asarray([item.get("distance", np.nan) for item in lane_matches], dtype=float)
    lane_directions = np.asarray([item.get("heading", np.nan) for item in lane_matches], dtype=float)
    heading_error = np.abs(np.asarray(wrap_angle(track.headings[indices] - lane_directions))) if lane_applies else np.asarray([])
    speed = np.asarray(track.velocities[indices], dtype=float) if lane_applies else np.asarray([])
    wrong_way = matched & np.isfinite(speed) & (speed > float(minimum_wrong_way_speed)) & (heading_error > float(wrong_way_threshold)) if lane_applies else np.asarray([])
    lane_boundary = provider.distance_to_lane_boundary(points) if lane_applies else np.asarray([])
    progress = provider.route_progress(points)
    outside = np.maximum(0.0, -signed)
    return {
        "status": "available",
        "agent_type": track.agent_type,
        "provider": provider.query_metadata(),
        "minimum_obb_occupancy_threshold": float(minimum_occupancy),
        "evaluated_indices": indices.tolist(),
        "evaluated_timestamps": track.timestamps[indices].tolist(),
        "evaluated_frame_count": int(len(indices)),
        "center_offroad_flags": center_offroad.tolist(),
        "footprint_offroad_flags": footprint_offroad.tolist(),
        "footprint_coverage_series": footprints.tolist(),
        "signed_distance_series": signed.tolist(),
        "center_offroad_ratio": float(np.mean(center_offroad)),
        "footprint_offroad_ratio": float(np.mean(footprint_offroad)),
        "drivable_area_violation_frames": int(np.sum(footprint_offroad)),
        "drivable_area_violation_duration": float(np.sum(footprint_offroad) * dt),
        "maximum_offroad_distance": float(np.max(outside)),
        "mean_offroad_distance": float(np.mean(outside[footprint_offroad])) if np.any(footprint_offroad) else 0.0,
        "lane_match_rate": float(np.mean(matched)) if lane_applies and lane_available else NOT_AVAILABLE,
        "lane_center_distance_mean": float(np.nanmean(lane_distances[matched])) if np.any(matched) else NOT_AVAILABLE,
        "lane_center_distance_p90": float(np.nanpercentile(lane_distances[matched], 90)) if np.any(matched) else NOT_AVAILABLE,
        "lane_boundary_distance_min": float(np.min(lane_boundary[matched & np.isfinite(lane_boundary)])) if np.any(matched & np.isfinite(lane_boundary)) else NOT_AVAILABLE,
        "road_boundary_distance_min": float(np.nanmin(road_boundary)) if len(road_boundary) else NOT_AVAILABLE,
        "wrong_way_ratio": float(np.mean(wrong_way[matched])) if np.any(matched) else NOT_AVAILABLE,
        "wrong_way_duration": float(np.sum(wrong_way) * dt) if lane_applies and lane_available else NOT_AVAILABLE,
        "lane_direction_error_mean": float(np.nanmean(heading_error[matched])) if np.any(matched) else NOT_AVAILABLE,
        "nearest_lane_ids": [item.get("lane_id") if item.get("status") == "matched" else None for item in lane_matches],
        "route_progress": progress.get("progress_m"),
        "route_completion_proxy": progress.get("completion"),
        "map_query_failure_rate": 0.0,
        # Stable aliases consumed by the pre-existing scorecard.
        "drivable_area_occupancy_ratio": float(np.mean(footprints)),
        "off_road_area_ratio": float(np.mean(1.0 - footprints)),
        "off_road_frame_ratio": float(np.mean(footprint_offroad)),
        "off_road_agent": bool(np.any(footprint_offroad)),
        "off_road_indices": indices[footprint_offroad].tolist(),
        "off_road_flags": footprint_offroad.tolist(),
        "obb_outside_distance_series": outside.tolist(),
        "maximum_obb_outside_distance": float(np.max(outside)),
        "lane_center_deviation": stats(lane_distances[matched].tolist()),
        "lane_heading_error": stats(heading_error[matched].tolist()),
    }


def evaluate_scene_map_provider(tracks, provider, config):
    per_agent = {track.agent_id: evaluate_track_map_provider(
        track, provider, float(config.get("minimum_obb_occupancy", 0.99)),
        float(config.get("wrong_way_heading_threshold_rad", math.pi / 2)),
        float(config.get("minimum_wrong_way_speed_mps", 0.5))) for track in tracks}
    available_all = [value for value in per_agent.values() if value.get("status") == "available"]
    available = [value for value in available_all if value.get("agent_type") in {"vehicle", "ego"}]
    def mean(key, rows=available):
        values = [item[key] for item in rows if isinstance(item.get(key), (int, float))]
        return float(np.mean(values)) if values else NOT_AVAILABLE
    def total(key, rows=available):
        values = [item[key] for item in rows if isinstance(item.get(key), (int, float))]
        return float(np.sum(values)) if values else NOT_AVAILABLE
    def extreme(key, function, rows=available):
        values = [item[key] for item in rows if isinstance(item.get(key), (int, float))]
        return float(function(values)) if values else NOT_AVAILABLE
    pedestrians = [value for value in available_all if value.get("agent_type") == "pedestrian"]
    bicycles = [value for value in available_all if value.get("agent_type") == "bicycle"]
    return {"status": "available" if available else NOT_AVAILABLE, "provider": provider.query_metadata(),
            "center_offroad_ratio": mean("center_offroad_ratio"), "footprint_offroad_ratio": mean("footprint_offroad_ratio"),
            "off_road_frame_ratio": mean("off_road_frame_ratio"), "off_road_area_ratio": mean("off_road_area_ratio"),
            "off_road_agent_ratio": float(np.mean([item["off_road_agent"] for item in available])) if available else NOT_AVAILABLE,
            "lane_match_rate": mean("lane_match_rate"), "lane_center_distance_mean": mean("lane_center_distance_mean"),
            "lane_center_distance_p90": mean("lane_center_distance_p90"),
            "lane_boundary_distance_min": extreme("lane_boundary_distance_min", min),
            "road_boundary_distance_min": extreme("road_boundary_distance_min", min),
            "wrong_way_ratio": mean("wrong_way_ratio"), "map_query_failure_rate": mean("map_query_failure_rate"),
            "wrong_way_duration": total("wrong_way_duration"),
            "drivable_area_violation_frames": total("drivable_area_violation_frames"),
            "drivable_area_violation_duration": total("drivable_area_violation_duration"),
            "maximum_offroad_distance": extreme("maximum_offroad_distance", max),
            "mean_offroad_distance": mean("mean_offroad_distance"),
            "route_progress": mean("route_progress"), "route_completion_proxy": mean("route_completion_proxy"),
            "vehicle_agent_count": len(available), "all_agent_count": len(available_all),
            "all_agent_center_offroad_ratio": mean("center_offroad_ratio", available_all),
            "all_agent_footprint_offroad_ratio": mean("footprint_offroad_ratio", available_all),
            "pedestrian_center_offroad_ratio": mean("center_offroad_ratio", pedestrians),
            "bicycle_center_offroad_ratio": mean("center_offroad_ratio", bicycles),
            "per_agent": per_agent}


def evaluate_scene_map_rules(tracks, map_path, config):
    if not map_path:
        return {"status": NOT_AVAILABLE, "reason": "no map file", "per_agent": {}}
    try:
        from driveaudit_eval.utils.map_provider.opendrive_provider import OpenDriveMapProvider
        provider = OpenDriveMapProvider.from_file(map_path, float(config.get("sample_step_m", 0.5)))
    except Exception as exc:
        return {"status": NOT_AVAILABLE, "reason": "%s: %s" % (type(exc).__name__, exc), "per_agent": {}}
    result = evaluate_scene_map_provider(tracks, provider, config)
    result.update({"map_file": str(map_path), "semantic_support": provider.map_data["semantic_support"], "_map_data": provider.map_data})
    return result


def compare_map_baseline(generated, ground_truth):
    gen = generated.get("per_agent", {})
    ref = ground_truth.get("per_agent", {})
    per_agent = {}
    for agent_id in sorted(set(gen) & set(ref)):
        a, b = gen[agent_id], ref[agent_id]
        if a.get("status") != "available" or b.get("status") != "available":
            per_agent[agent_id] = {"status": NOT_AVAILABLE, "reason": "map metric unavailable for generated or reference"}
            continue
        ga = dict(zip((round(float(t), 9) for t in a["evaluated_timestamps"]), a["off_road_flags"]))
        gb = dict(zip((round(float(t), 9) for t in b["evaluated_timestamps"]), b["off_road_flags"]))
        common = sorted(set(ga) & set(gb))
        shared = sum(bool(ga[t] and gb[t]) for t in common)
        new = sum(bool(ga[t] and not gb[t]) for t in common)
        resolved = sum(bool(not ga[t] and gb[t]) for t in common)
        denominator = max(1, len(common))
        per_agent[agent_id] = {
            "status": "available",
            "generated_offroad_ratio": a["off_road_frame_ratio"],
            "gt_offroad_ratio": b["off_road_frame_ratio"],
            "shared_offroad_ratio": float(shared / denominator),
            "new_offroad_ratio": float(new / denominator),
            "resolved_offroad_ratio": float(resolved / denominator),
            "offroad_gap": float(a["off_road_frame_ratio"] - b["off_road_frame_ratio"]),
            "shared_offroad_timestamps": [t for t in common if ga[t] and gb[t]],
            "new_offroad_timestamps": [t for t in common if ga[t] and not gb[t]],
            "resolved_offroad_timestamps": [t for t in common if not ga[t] and gb[t]],
            "comparison_frames": len(common),
        }
    available = [item for item in per_agent.values() if item.get("status") == "available"]
    return {
        "status": "available" if available else NOT_AVAILABLE,
        "per_agent": per_agent,
        "generated_offroad_ratio": float(np.mean([item["generated_offroad_ratio"] for item in available])) if available else NOT_AVAILABLE,
        "gt_offroad_ratio": float(np.mean([item["gt_offroad_ratio"] for item in available])) if available else NOT_AVAILABLE,
        "shared_offroad_ratio": float(np.mean([item["shared_offroad_ratio"] for item in available])) if available else NOT_AVAILABLE,
        "new_offroad_ratio": float(np.mean([item["new_offroad_ratio"] for item in available])) if available else NOT_AVAILABLE,
        "resolved_offroad_ratio": float(np.mean([item["resolved_offroad_ratio"] for item in available])) if available else NOT_AVAILABLE,
        "offroad_gap": float(np.mean([item["offroad_gap"] for item in available])) if available else NOT_AVAILABLE,
    }
