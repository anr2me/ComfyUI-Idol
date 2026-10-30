"""Use a FLUX.2 VAE file from ComfyUI's models/vae folder (for example flux2-vae.safetensors) with Idol.

Idol's image encoder loads the VAE with diffusers: AutoencoderKLFlux2.from_pretrained(folder, subfolder="vae").
This module builds such a folder around the single file. It writes the model config and links the weights, so
the 336 MB file is not copied when the file system allows a link.

The file must contain every tensor that the model needs. diffusers does not fail on missing tensors (it
initializes them at random), so the check here is strict: a file with another key layout raises an error.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil

import torch
from safetensors import safe_open

log = logging.getLogger("ComfyUI-Idol")

# Config of diffusers' AutoencoderKLFlux2 (the same in the FLUX.2 dev and FLUX.2 klein repositories).
VAE_CONFIG = {
    "act_fn": "silu",
    "batch_norm_eps": 0.0001,
    "batch_norm_momentum": 0.1,
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 32,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "patch_size": [2, 2],
    "sample_size": 1024,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "use_post_quant_conv": True,
    "use_quant_conv": True,
}

# Used only if the expected key list cannot be built from diffusers.
FALLBACK_KEYS = {
    "encoder.conv_in.weight",
    "encoder.down_blocks.0.resnets.0.norm1.weight",
    "encoder.mid_block.attentions.0.group_norm.weight",
    "decoder.conv_in.weight",
    "decoder.up_blocks.3.resnets.2.conv2.weight",
    "quant_conv.weight",
    "post_quant_conv.weight",
    "bn.running_mean",
    "bn.running_var",
}


def _expected_keys() -> set[str]:
    from diffusers.models import AutoencoderKLFlux2
    with torch.device("meta"):  # no memory is used: only the names are needed
        model = AutoencoderKLFlux2.from_config(VAE_CONFIG)
    return set(model.state_dict().keys())


def check_keys(path: str) -> None:
    """Raise ValueError unless the file holds every tensor of the FLUX.2 VAE."""
    with safe_open(path, framework="pt", device="cpu") as f:
        have = set(f.keys())
    try:
        expected = _expected_keys()
    except Exception as error:
        log.warning("Could not build the expected FLUX.2 VAE key list (%s). Checking a short key list instead.", error)
        expected = FALLBACK_KEYS
    missing = sorted(k for k in expected - have if not k.endswith("num_batches_tracked"))
    if missing:
        raise ValueError(
            f"'{os.path.basename(path)}' is not a FLUX.2 VAE in the diffusers key layout: {len(missing)} tensors are "
            f"missing (for example {', '.join(missing[:3])}). Use flux2-vae.safetensors from Comfy-Org/flux2-dev, "
            "or give flux2_vae_source a diffusers folder that contains vae/.")


def _link_or_copy(source: str, target: str) -> None:
    if os.path.lexists(target):
        os.remove(target)
    for method in (os.symlink, os.link, shutil.copyfile):  # symlink needs a privilege on some Windows setups
        try:
            method(source, target)
            return
        except OSError:
            continue
    raise OSError(f"Could not link or copy {source} to {target}")


def prepare_flux2_vae_folder(path: str, cache_root: str) -> str:
    """Return a folder that holds vae/config.json and vae/diffusion_pytorch_model.safetensors for this file."""
    path = os.path.realpath(path)
    info = os.stat(path)
    tag = hashlib.sha1(f"{path}|{info.st_size}|{info.st_mtime_ns}".encode()).hexdigest()[:12]
    root = os.path.join(cache_root, tag)
    vae_dir = os.path.join(root, "vae")
    weights = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
    config = os.path.join(vae_dir, "config.json")
    if os.path.isfile(weights) and os.path.isfile(config):
        return root

    check_keys(path)
    os.makedirs(vae_dir, exist_ok=True)
    with open(config, "w") as f:
        json.dump({"_class_name": "AutoencoderKLFlux2", **VAE_CONFIG}, f, indent=2)
    _link_or_copy(path, weights)
    log.info("Using the FLUX.2 VAE file %s", path)
    return root
