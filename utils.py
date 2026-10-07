"""Shared helpers for CapField-OPD.

Low-level tensor utilities (latent packing / position ids) and a small
parameter-count logger. No project-specific state is kept here.
"""

import os

import torch


def unwrap(model):
    """Return the raw module when `model` is wrapped by DDP."""
    return model.module if hasattr(model, "module") else model


def assert_valid_sequence(step_indices, num_denoising_steps):
    """Distillation timesteps must be strictly increasing and inside the schedule."""
    if not step_indices:
        return
    assert 0 <= step_indices[0] and step_indices[-1] < num_denoising_steps, (
        f"train timesteps must lie in [0, {num_denoising_steps}), "
        f"got [{step_indices[0]}, {step_indices[-1]}]"
    )
    for a, b in zip(step_indices, step_indices[1:]):
        assert a < b, f"train timesteps must be strictly increasing, got {a} -> {b}"


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    """[B, C, H, W] -> [B, (H/2)*(W/2), 4C] (FLUX patchify)."""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


def unpack_latents(latents, height, width, vae_scale_factor):
    """[B, (H/2)*(W/2), 4C] -> [B, C', H', W'] (inverse of `pack_latents`)."""
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // (2 * 2), height, width)


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """FLUX image position ids for a latent grid of (height, width) patches."""
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(height)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(width)[None, :]
    h, w, c = ids.shape
    return ids.reshape(h * w, c).to(device=device, dtype=dtype)


def log_trainable_parameters(model, log_file):
    """Write every trainable parameter (and a summary) to `log_file`."""
    total_params = 0
    trainable_params = 0
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== Trainable parameters ===\n")
        f.write(f"Model: {type(model).__name__}\n\n")
        for name, param in model.named_parameters():
            num = param.numel()
            total_params += num
            if param.requires_grad:
                trainable_params += num
                f.write(f"{name}\n  shape={list(param.shape)} dtype={param.dtype} count={num}\n")
        f.write("\n=== Summary ===\n")
        f.write(f"Total params:     {total_params:,}\n")
        f.write(f"Trainable params: {trainable_params:,}\n")
        f.write(f"Trainable ratio:  {100 * trainable_params / max(total_params, 1):.4f}%\n")
