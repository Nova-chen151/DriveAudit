"""Resolve native checkpoint/environment profiles without an earlier experiment."""

ENVIRONMENTS = {
    "basedreamer": "wd_basedreamer",
    "dreamforge": "wd_dreamforge",
    "horizondrive": "horizondrive",
    "magicdrive": "magicdrive_t",
    "opendwm": "opendwm",
    "panacea": "panacea",
}
DRIVEAUDIT_SD = "base-pretrained/pretrained/stable-diffusion-v1-5"
WEIGHTS = {
    "basedreamer": {"render_checkpoint": "base-pretrained/dreamer_pretrained/SDv1.5_mv_single_ref_nus/weight-S200000",
                    "sd15": DRIVEAUDIT_SD},
    "dreamforge": {"render_checkpoint": "models--Jianbiao--dreamforge-t",
                   "single_checkpoint": "models--Jianbiao--dreamforge-s", "sd15": DRIVEAUDIT_SD},
    "horizondrive": {"render_checkpoint": "models--zcliangyue--HorizonDrive/horizondrive-dit.safetensors",
                     "wan_checkpoint": "models--Wan-AI--Wan2.1-T2V-1.3B"},
    "magicdrive": {"render_checkpoint": "models--flymin--MagicDrive-t-60f-224x400-80k",
                   "sd15": "models--stable-diffusion-v1-5--stable-diffusion-v1-5"},
    "opendwm": {"render_checkpoint": "models--wzhgba--opendwm-models/ctsd_unimlvg_tirda_bm_nwa_60k.pth",
                "base_model": "models--stabilityai--stable-diffusion-3-medium-diffusers"},
    "panacea": {"render_checkpoint": "models--orangewen--panacea_ckpts/panaceaplus_40k_deepspeed.ckpt",
                "clip_checkpoint": "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"},
}


def model_settings(environments, checkpoints):
    native = {}
    for model, weights in WEIGHTS.items():
        settings = {name: str(checkpoints / relative) for name, relative in weights.items()}
        settings["python"] = {"renderer": str(environments / ENVIRONMENTS[model] / "bin/python")}
        native[model] = settings
    return native
