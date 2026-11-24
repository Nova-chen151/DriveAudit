"""Observed-support DAC: unknown map samples are neither road nor off-road."""
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

DRIVABLE, NON_DRIVABLE, UNKNOWN_MAP = 1, 0, -1
MIN_MAP_COVERAGE = .8


@dataclass
class CanonicalMapContext:
    drivable_area: np.ndarray
    observed_area: np.ndarray
    extent: np.ndarray
    resolution: float
    route_polyline: np.ndarray
    metadata: dict
    lane_direction_if_available: object = None

    def classify(self, points):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Map queries require finite Nx2 points')
        ij = np.floor((points-self.extent[[0, 2]])/self.resolution).astype(int)
        inside = ((ij >= 0) & (ij < self.observed_area.shape)).all(axis=1)
        result = np.full(len(points), UNKNOWN_MAP, dtype=np.int8)
        indices = np.flatnonzero(inside)
        observable = self.observed_area[ij[indices, 0], ij[indices, 1]]
        indices = indices[observable]
        result[indices] = self.drivable_area[ij[indices, 0], ij[indices, 1]].astype(np.int8)
        return result

    def dac_common(self, centers, headings, length, width, min_coverage=.8, sample_spacing_m=.25):
        """Uniform footprint interior samples, same metric for every source map.

        0.25 m samples resolve the existing 0.5 m maps. The frozen provisional
        coverage floor requires at least 80% actual map support; no interpolation,
        dilation or route-derived labels turn unknown cells into road.
        """
        centers, headings = np.asarray(centers, float), np.asarray(headings, float)
        if (centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0 or
            headings.shape != (len(centers),) or not np.isfinite(centers).all() or
            not np.isfinite(headings).all() or not np.isfinite([length,width,sample_spacing_m,min_coverage]).all() or
            min(length,width,sample_spacing_m) <= 0 or not MIN_MAP_COVERAGE <= min_coverage <= 1):
            raise ValueError('Invalid footprint or coverage specification')
        nx, ny = np.ceil(np.array([length, width])/sample_spacing_m).astype(int)
        local = np.stack(np.meshgrid((np.arange(nx)+.5)*length/nx-length/2,
                                     (np.arange(ny)+.5)*width/ny-width/2, indexing='ij'), axis=-1).reshape(-1,2)
        cosine, sine = np.cos(headings), np.sin(headings)
        rotation = np.stack([cosine,-sine,sine,cosine],axis=-1).reshape(-1,2,2)
        points = np.einsum('nij,kj->nki', rotation, local)+centers[:,None,:]
        states = self.classify(points.reshape(-1,2)).reshape(len(centers),-1)
        observed = int(np.sum(states != UNKNOWN_MAP)); drivable = int(np.sum(states == DRIVABLE))
        coverage = observed/states.size
        valid = observed > 0 and coverage >= min_coverage
        step_observed = np.sum(states != UNKNOWN_MAP, axis=1)
        step_drivable = np.sum(states == DRIVABLE, axis=1)
        return dict(protocol='observed_common_v1', DAC_common=drivable/observed if valid else None,
                    DAC=drivable/observed if valid else None, dac_map_coverage=coverage,
                    drivable_samples=drivable, observable_samples=observed, total_samples=int(states.size),
                    unknown_samples=int(np.sum(states == UNKNOWN_MAP)), non_drivable_samples=int(np.sum(states == NON_DRIVABLE)),
                    per_step_coverage=np.mean(states != UNKNOWN_MAP,axis=1).tolist(),
                    per_step_drivable_fraction=[float(road/count) if count else None for road,count in zip(step_drivable,step_observed)],
                    coverage_threshold=min_coverage, sample_spacing_m=sample_spacing_m,
                    status='valid' if valid else 'insufficient_map_coverage',
                    reason=None if valid else 'insufficient_observed_drivable_area_map_support')


def load_map_context(scene):
    scene = Path(scene); folder = scene/'map'
    meta = json.loads((folder/'map_meta.json').read_text())
    if meta.get('array_axes') != 'channel,x_forward,y_left' or meta.get('coordinate_frame') != 'history_last_ego_x_forward_y_left_meters':
        raise ValueError('Canonical map requires fixed anchor XY frame')
    with np.load(folder/'semantic_map.npz', allow_pickle=False) as data:
        road = data['mask'][meta['channels'].index('drivable_area')].astype(bool)
        if 'drivable_area' not in meta.get('available_channels', []):
            raise ValueError('Native road semantics unavailable')
        observed = data['observed_surface'].astype(bool) if 'observed_surface' in data else np.ones_like(road, bool)
    resolution = float(meta['resolution_m']); extent = np.asarray(meta['extent_m'],float)
    if (resolution <= 0 or observed.shape != road.shape or extent.shape != (4,) or
        not np.allclose(np.asarray(road.shape)*resolution, extent[[1,3]]-extent[[0,2]])):
        raise ValueError('Invalid raster extent or observed-area dimensions')
    pose_file = scene/'calibration/reference_ego_poses.json'
    route = np.asarray(json.loads(pose_file.read_text())['ego_to_anchor'])[:,:2,3] if pose_file.is_file() else np.empty((0,2))
    return CanonicalMapContext(road, observed, extent, resolution, route,
        dict(source=str(folder/'semantic_map.npz'), source_manifest=meta.get('source_manifest'),
             map_semantics='native labelled road surface; unknown cells excluded',
             source_status=meta.get('status'), route_used_for_road_labels=False,
             coverage_threshold_policy='provisional fixed 0.8; audit distributions, never lower to force passing'))
