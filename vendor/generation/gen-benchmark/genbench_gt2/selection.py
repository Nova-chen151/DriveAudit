"""Generation-set selection and deterministic modulo sharding."""

from typing import List, Optional, Sequence

from .constants import PILOT_INDICES
from .schema import SceneRef


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{} must be an integer".format(name))
    return value


def select_scenes(scenes: Sequence[SceneRef], pilot: bool, scene_indices: Optional[Sequence[int]], limit: Optional[int]) -> List[SceneRef]:
    """Select scenes in generation-set order, optionally using pilot indices."""
    if pilot and scene_indices is not None:
        raise ValueError("pilot and explicit scene indices are mutually exclusive")
    if limit is not None:
        _integer(limit, "limit")
        if limit <= 0:
            raise ValueError("limit must be positive")

    ordered = list(scenes)
    by_index = {}
    for scene in ordered:
        if not isinstance(scene, SceneRef):
            raise ValueError("scenes must contain SceneRef values")
        if scene.index in by_index:
            raise ValueError("scenes contain duplicate index {}".format(scene.index))
        by_index[scene.index] = scene

    if isinstance(scene_indices, str):
        scene_indices = tuple(int(value.strip()) for value in scene_indices.split(",") if value.strip())
    if scene_indices is not None:
        requested = []
        for index in scene_indices:
            _integer(index, "scene index")
            if index in requested:
                raise ValueError("duplicate scene index {}".format(index))
            if index not in by_index:
                raise ValueError("scene index {} is out of range".format(index))
            requested.append(index)
        selected = [scene for scene in ordered if scene.index in set(requested)]
    elif pilot:
        pilot_indices = set(PILOT_INDICES)
        selected = [scene for scene in ordered if scene.index in pilot_indices]
    else:
        selected = ordered
    return selected[:limit] if limit is not None else selected


def shard_scenes(scenes: Sequence[SceneRef], rank: int, world_size: int) -> List[SceneRef]:
    """Return scenes assigned to ``rank`` using stable scene-index modulo."""
    _integer(rank, "rank")
    _integer(world_size, "world_size")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    return [scene for scene in scenes if scene.index % world_size == rank]


__all__ = ["select_scenes", "shard_scenes"]
