"""Required released assets and primary config used by stock service launchers."""
from .runtime import asset_root, model_root

POLICIES = {
    "drivor": ("DrivoR", "navsim/planning/script/config/common/agent/drivoR.yaml",
               ("weights/drivor_Nav1_25epochs.pth", "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors")),
    "uniad": ("UniAD", "projects/configs/stage2_e2e/base_e2e.py",
              ("ckpts/uniad_base_e2e.pth", "ckpts/CAMixerSRx4_DF.pth", "data_temp/data_template.pth",
               "data_temp/others/motion_anchor_infos_mode6.pkl")),
    "vad": ("VAD", "projects/configs/VAD/VAD_base_stage_2.py",
            ("ckpts/VAD_base.pth", "ckpts/CAMixerSRx4_DF.pth", "data_temp/data_template.pth")),
    "sparsedrivev2": ("SparseDriveV2", "navsim/agents/sparsedrive/sparsedrive_config.py",
                      ("ckpt/sparsedrive_navsimv2_90p3.ckpt", "ckpt/resnet34.bin",
                       "ckpt/kmeans/path_1024.npy", "ckpt/kmeans/velocity_256.npy", "ckpt/kmeans/trajectory_1024_256.npz")),
    "diffusiondrivev2": ("DiffusionDriveV2", "navsim/agents/diffusiondrivev2/diffusiondrivev2_sel_config.py",
                         ("ckpts/diffusiondrivev2_sel.ckpt", "ckpts/resnet34.a1_in1k/pytorch_model.bin", "kmeans_navsim_traj_20.npy")),
    "goalflow": ("GoalFlow", "navsim/agents/goalflow/goalflow_config.py",
                 ("data/goalflow_navi_epoch_99-step_132500.ckpt", "data/goalflow_traj_epoch_54-step_18260.ckpt",
                  "data/cluster_points_8192_.npy", "data/weights_manifest.json")),
}


def resources(policy):
    model, configuration, names = POLICIES[policy]
    return model_root(model) / configuration, {name: asset_root(model) / name for name in names}
