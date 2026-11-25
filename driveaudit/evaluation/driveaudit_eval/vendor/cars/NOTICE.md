# CARS attribution and modification notice

DriveAudit uses the collision geometry, FSM, CC-JP and RSS controllers, and finite-horizon replay from `scripts/evaluate/replay.py` in *CARS: Learning Responsibility-Attributed Adversarial Scenarios for Testing Autonomous Vehicles*. [Upstream repository](https://github.com/RoboSafe-Lab/CARS-code) · [Apache License 2.0](LICENSE)

`responsibility_models.py` contains 15 definitions extracted from the original file (SHA-256 `0a3a248e7eeb77e67943edcab615d9b20508f788b87a27e75ce55acd3414389d`). The extraction omits the command-line interface, data loading and map ingestion.

The retained source comments credit Mattas et al. (2022), UN R157 Annex 4 Appendix 3, Shalev-Shwartz et al. (2017) and the JRC-FSM reference implementation.
