# Why this code is retained

`driveaudit_eval` responsibility evaluation invokes FSMController, CCJPLaneAwareController, RSSController, check_raw_collision, replay_episode and compute_dist_to_react. Their transitive dependencies are retained in one module.

Only NumPy and the Python abc standard-library module are imported. No CARS checkpoint, CTG, Pplan, trajdata, h5py, tqdm, nuScenes installation or CARS CLI is required by this subset. Native lane measurements are supplied by DriveAudit.

Full CARS source can be externalized or removed after adapter paths are switched and numerical regression passes. This subset can be externalized later only if an equivalently versioned dependency exposes these definitions.
