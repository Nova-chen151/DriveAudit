# DriveAudit challenge: generate future traffic and six-camera RGB

[![English](https://img.shields.io/badge/English-2878B5?style=flat-square)](challenge.md) [![Simplified Chinese](https://img.shields.io/badge/Simplified_Chinese-777777?style=flat-square)](challenge.zh-CN.md)

For each of the 250 nuScenes scenes, predict the future motion of the traffic participants, then render that prediction from six cameras on the ego vehicle. Use the supplied future ego and camera poses for rendering. Both outputs must describe the same scene at the same times. Use the provided history and conditioning data without looking up the withheld future traffic or images.

## Inputs

The repository expects `work/resources/participant.tar.gz`. Extracting the archive creates `participant/`. Its `dataset.json` lists the official scene IDs. Each scene has ten history frames (0–9) and 80 target frames (10–89).

| Path under `participant/scenes/<scene_id>/` | Contents |
| --- | --- |
| `history/rgb/<camera>/`, `history/agents.csv`, `history/ego.csv` | Six-view RGB, anonymous traffic tracks and ego history |
| `calibration/`, `map/` | Camera calibration, historical poses and static map information |
| `conditioning/ego_trajectory.csv`, `conditioning/ego_poses.json`, `conditioning/camera_poses.json` | Given ego motion and camera poses |
| `timeline.json`, `task.json` | Sampling times, target frames, camera names and image size |

Read the timestamps from `timeline.json`. Use the fixed ego coordinate system at history frame 9 throughout: x points forward, y left and z up.

## What to submit

Submit one trajectory CSV and 480 generated RGB JPEGs per scene. The directory structure is shown below for the first scene. Repeat it for every ID in `dataset.json`. Use regular files.

```text
submission/
  scenes/
    case_000001/
      future/
        future_trajectories.csv
        rgb/
          CAM_FRONT_LEFT/    000010.jpg … 000089.jpg
          CAM_FRONT/         000010.jpg … 000089.jpg
          CAM_FRONT_RIGHT/   000010.jpg … 000089.jpg
          CAM_BACK_LEFT/     000010.jpg … 000089.jpg
          CAM_BACK/          000010.jpg … 000089.jpg
          CAM_BACK_RIGHT/    000010.jpg … 000089.jpg
```

**Traffic trajectories.** Use this exact header and column order in `future_trajectories.csv`:

```csv
frame_index,agent_id,category,is_dynamic,x,y,z,width,length,height,yaw,vx,vy,valid
```

- Keep each historical actor’s anonymous ID and category. Write one row per actor per target frame, including `valid=False` rows when the actor is absent. For a newly generated actor, use a scene-local ID such as `new_0001` and provide one row per frame from its first appearance through frame 89.
- Use metres for position and dimensions, radians for yaw and metres per second for velocity. All numeric values must be finite, including on invalid rows. Width, length and height must be positive. Use boolean values for `is_dynamic` and `valid`.
- For new actors, the supported categories are `car`, `vehicle`, `truck`, `bus`, `trailer`, `construction_vehicle`, `motorcycle`, `bicycle`, `pedestrian`, `animal`, `traffic_cone`, `barrier` and `other`. Keep the category consistent across frames.

**RGB frames.** Write one 400 × 224 RGB JPEG per camera per target frame. Use the six-digit frame number from the trajectory CSV as the file stem, including leading zeros.

## Check your submission

From the directory containing the extracted `participant/` and your `submission/`, run the following commands with Python 3, NumPy and Pillow installed.

```bash
# Check one scene after predicting traffic.
python participant/validate_submission.py --inputs participant --submission submission \
  --scene case_000001 --trajectories-only

# Optionally prepare the same scene for a DriveAudit renderer.
python participant/prepare_render_inputs.py --inputs participant --trajectories submission \
  --scene case_000001 --output render_inputs

# After rendering all scenes, check the complete submission.
python participant/validate_submission.py --inputs participant --submission submission
```

A successful final check prints `FORMAT_PASS 250 scenes` followed by a reminder that it checks file format. Submit the contents shown under `submission/` above. The input package and intermediate render inputs stay with you.
