# CARS attribution and modification notice

Upstream: https://github.com/RoboSafe-Lab/CARS-code

Project: CARS: Learning Responsibility-Attributed Adversarial Scenarios for Testing Autonomous Vehicles.

License: Apache License 2.0; the complete original LICENSE is retained byte-for-byte. No separate upstream NOTICE file was present in the supplied repository. The upstream LICENSE appendix retains its original placeholder copyright line; no copyright owner has been invented or replaced.

Source: `scripts/evaluate/replay.py`. Original file SHA-256: `0a3a248e7eeb77e67943edcab615d9b20508f788b87a27e75ce55acd3414389d`.

Modification: extracted only the transitive function/class closure used by DriveAudit responsibility evaluation. The 15 retained definitions, including their docstrings and comments, are copied verbatim; AST equality is verified. Removed module-level CLI, HDF5, progress-bar and native-map ingestion dependencies. No controller formula, parameter, state machine, geometry or replay behavior was changed.

The original source credits Mattas et al. (2022), UN R157 Annex 4 Appendix 3, Shalev-Shwartz et al. (2017), and consultation of the JRC-FSM reference implementation. These credits remain in the retained source comments.

This subset is a counterfactual diagnostic; its buffered OBB geometry is not the original Agent SAT collision trigger and does not replace that trigger.
