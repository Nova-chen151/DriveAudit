# Participant inputs

Copy or symlink the downloaded nuScenes participant archive to `work/resources/participant.tar.gz`. Keep it compressed. The runner extracts the selected scenes. To use the archive from another location, set `archive` in [generation.json](../../configs/generation.json).

Use `envs/` for the model-specific Python environments or symlinks to them. Organizers running paired evaluation also need `scene_mapping.json` here and access to the recorded reference data.

See the [generation guide](../../docs/generation.md) for the commands to generate traffic trajectories and six-camera video.
