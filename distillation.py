"""Rollout, coordinate-conditioned teacher target and OPD loss.

Per step: rollout the student from noise, build the teacher velocity for the
prompt's coordinate, take the MSE, and back-propagate each timestep at once.
"""

import os
import re

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from utils import pack_latents, prepare_latent_image_ids, unwrap

_VANILLA = "vanilla"


@torch.no_grad()
def sample_student_trajectory(
    args,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    guidance,
    sigma_scheduler,
    autocast,
    sources,
    task_hook=None,
    rollout_pools=None,
    rollout_probs=None,
):
    """Student ODE from noise; the coordinate is sampled once and reused for all steps."""
    device = encoder_hidden_states.device
    dtype = encoder_hidden_states.dtype
    batch_size = encoder_hidden_states.shape[0]

    noise = torch.randn(
        (batch_size, 16, args.height // 8, args.width // 8),
        device=device,
        dtype=dtype,
    )
    image_latents = pack_latents(noise, batch_size, 16, args.height // 8, args.width // 8)
    latent_image_ids = prepare_latent_image_ids(
        batch_size, args.height // 16, args.width // 16, device, dtype
    )

    if text_ids.ndim == 3:
        txt_ids = text_ids[0]
    else:
        txt_ids = text_ids[0:1].expand(encoder_hidden_states.shape[1], -1)

    all_latents = [image_latents]

    if task_hook is not None and rollout_pools is not None:
        task_hook.configure_rollout(rollout_pools, rollout_probs)
    unwrap(transformer).set_adapter("student")
    if task_hook is not None:
        task_hook.enable()
        task_hook.set_rollout_batch(sources)

    for sigma_idx in range(len(sigma_scheduler) - 1):
        timestep = sigma_scheduler[sigma_idx].expand(batch_size).to(dtype)
        with autocast():
            pred_velocity = transformer(
                hidden_states=image_latents,
                timestep=timestep,
                guidance=guidance.expand(batch_size),
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
        image_latents = image_latents.float() - (
            sigma_scheduler[sigma_idx] - sigma_scheduler[sigma_idx + 1]
        ) * pred_velocity.float()
        image_latents = image_latents.to(dtype)
        all_latents.append(image_latents)

    if task_hook is not None:
        task_hook.disable()
    return all_latents, latent_image_ids


@torch.no_grad()
def _compute_teacher_velocity(
    transformer,
    anchor_latent,
    pooled_prompt_embeds,
    encoder_hidden_states,
    txt_ids,
    latent_image_ids,
    guidance,
    timestep,
    sources,
    autocast,
    coord_matrix,
    registry,
):
    """Teacher velocity for every sample, mixed from its coordinate's teachers."""

    def _fwd(sel):
        with autocast():
            return transformer(
                hidden_states=anchor_latent[sel],
                timestep=timestep[sel],
                guidance=guidance.expand(anchor_latent[sel].shape[0]),
                pooled_projections=pooled_prompt_embeds[sel],
                encoder_hidden_states=encoder_hidden_states[sel],
                txt_ids=txt_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]

    assert registry is not None, "teacher velocity needs args.teacher_registry"

    if coord_matrix is None:
        src_list = [int(s) for s in sources.tolist()]
        dim = len(registry.entries[0]["vec"]) if registry.entries else max(src_list) + 1
        rows = []
        for s in src_list:
            e = [0.0] * dim
            e[int(s)] = 1.0
            rows.append(e)
    else:
        rows = coord_matrix.detach().to(device="cpu", dtype=torch.float32).tolist()

    # Group samples by adapter so each adapter is forwarded only once.
    groups = {}
    for i, row in enumerate(rows):
        for name, weight in registry.decompose(row):
            g_idx, g_w = groups.setdefault(name, ([], []))
            g_idx.append(i)
            g_w.append(float(weight))

    shape_w = (-1,) + (1,) * (anchor_latent.ndim - 1)
    device = anchor_latent.device
    velocity = torch.zeros_like(anchor_latent, dtype=torch.float32)
    for adapter_name, (g_idx, g_w) in groups.items():
        sel = torch.tensor(g_idx, device=device, dtype=torch.long)
        wt = torch.tensor(g_w, device=device, dtype=torch.float32)
        unwrap(transformer).set_adapter(adapter_name)
        velocity[sel] += wt.view(shape_w) * _fwd(sel).float()
    return velocity


def compute_opd_loss(
    args,
    transformer,
    all_latents,
    latent_image_ids,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    guidance,
    sigma_scheduler,
    sources,
    autocast,
    step_indices,
    task_hook=None,
    rollout_cond=None,
    loss_target="velocity",
    backward_fn=None,
    ddp_model=None,
):
    """Per-timestep MSE between the student and the coordinate's teacher target.

    Each timestep is back-propagated immediately; only the last one triggers the
    DDP all-reduce (equivalent to a single sum-then-backward).
    """
    device = encoder_hidden_states.device
    dtype = encoder_hidden_states.dtype
    batch_size = encoder_hidden_states.shape[0]

    if text_ids.ndim == 3:
        txt_ids = text_ids[0]
    else:
        txt_ids = text_ids[0:1].expand(encoder_hidden_states.shape[1], -1)

    per_step_losses = []
    num_steps = len(step_indices)
    for step_i, stepidx in enumerate(step_indices):
        anchor_latent = all_latents[stepidx]
        dt = sigma_scheduler[stepidx] - sigma_scheduler[stepidx + 1]
        timestep = sigma_scheduler[stepidx].expand(batch_size).to(dtype)

        # Per-timestep coordinate, drawn from levels matching the rollout support
        # (fallback to the full per-source pool when no rollout coordinate is given).
        step_coord = None
        if task_hook is not None:
            if rollout_cond is not None:
                task_hook.set_batch_from_rollout(sources, rollout_cond)
            else:
                task_hook.set_batch_from_sources(sources)
            step_coord = task_hook.last_cond_matrix

        # Teacher never sees the coordinate conditioning itself.
        if task_hook is not None:
            task_hook.disable()
        teacher_v = _compute_teacher_velocity(
            transformer,
            anchor_latent,
            pooled_prompt_embeds,
            encoder_hidden_states,
            txt_ids,
            latent_image_ids,
            guidance,
            timestep,
            sources,
            autocast,
            coord_matrix=step_coord,
            registry=getattr(args, "teacher_registry", None),
        )

        # Student forward (with gradients), same coordinate.
        unwrap(transformer).set_adapter("student")
        if task_hook is not None:
            task_hook.enable()
        with autocast():
            student_v = transformer(
                hidden_states=anchor_latent,
                timestep=timestep,
                guidance=guidance.expand(batch_size),
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
        student_v = student_v.float()

        if loss_target == "velocity":
            student_tgt, teacher_tgt = student_v, teacher_v
        elif loss_target == "pred_x0":
            sigma = sigma_scheduler[stepidx].float()
            student_tgt = anchor_latent.float() - sigma * student_v
            teacher_tgt = anchor_latent.float() - sigma * teacher_v
        else:  # next_latent
            student_tgt = anchor_latent.float() - dt * student_v
            teacher_tgt = anchor_latent.float() - dt * teacher_v

        loss_step = F.mse_loss(student_tgt, teacher_tgt)

        if backward_fn is not None:
            per_step_losses.append(loss_step.detach())
            is_last = step_i == num_steps - 1
            scaled = loss_step / num_steps
            if is_last or ddp_model is None or not hasattr(ddp_model, "no_sync"):
                backward_fn(scaled)
            else:
                with ddp_model.no_sync():
                    backward_fn(scaled)
        else:
            per_step_losses.append(loss_step)

    if task_hook is not None:
        task_hook.disable()

    loss = sum(per_step_losses) / len(per_step_losses)
    return loss, per_step_losses


def save_student_lora(transformer, save_dir, main_print, task_hook=None):
    """Save the student adapter and conditioner into `save_dir` (flat layout)."""
    os.makedirs(save_dir, exist_ok=True)
    unwrapped = transformer
    if hasattr(transformer, "module"):
        unwrapped = transformer.module

    unwrapped.set_adapter("student")
    unwrapped.save_pretrained(save_dir, safe_serialization=True)

    for name in sorted(os.listdir(save_dir)):
        path = os.path.join(save_dir, name)
        if os.path.isdir(path) and (name.startswith("teacher_") or name == _VANILLA):
            import shutil

            shutil.rmtree(path)

    # PEFT writes the active adapter into a "student/" subfolder; flatten it.
    student_subdir = os.path.join(save_dir, "student")
    if os.path.isdir(student_subdir):
        import shutil

        for name in os.listdir(student_subdir):
            shutil.move(os.path.join(student_subdir, name), os.path.join(save_dir, name))
        shutil.rmtree(student_subdir)

    for readme in (os.path.join(save_dir, "README.md"),):
        if os.path.exists(readme):
            os.remove(readme)

    adapter_st = os.path.join(save_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_st):
        _strip_extra_adapter_weights(adapter_st, main_print)

    if task_hook is not None:
        task_hook.save(save_dir)
        main_print(f"  Saved conditioner to {save_dir}/task_conditioner.pt")

    main_print(f"Saved student LoRA to {save_dir}")


def _strip_extra_adapter_weights(adapter_st_path, main_print=None):
    """Drop dead `*.lora_[AB].(teacher_N|vanilla).weight` keys from the safetensors."""
    junk = re.compile(r"\.lora_[AB]\.(teacher_\d+|vanilla)\.weight$")
    state_dict = load_file(adapter_st_path)
    cleaned = {k: v for k, v in state_dict.items() if not junk.search(k)}
    n_removed = len(state_dict) - len(cleaned)
    if n_removed > 0:
        save_file(cleaned, adapter_st_path)
        size_mb = os.path.getsize(adapter_st_path) / 1e6
        msg = f"  Stripped {n_removed} dead teacher tensors ({len(cleaned)} kept, {size_mb:.0f}MB)"
        (main_print or print)(msg)
    return n_removed
