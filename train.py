"""CapField-OPD training entry point.

Distills frozen task teachers (each a LoRA) into one student LoRA conditioned
on a capability-field coordinate. Per step:

    rollout -> build the coordinate's teacher target -> MSE loss -> update

No reward model is used (evaluation only, see reward_server.py).
"""

import argparse
import gc
import os
import time
from datetime import datetime

os.environ["no_proxy"] = "localhost,127.0.0.1"

import math

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from dataset import build_dataloader
from distillation import compute_opd_loss, sample_student_trajectory, save_student_lora
from ema import EMAModuleWrapper
from modeling import CapFieldHook, TeacherRegistry, load_model_with_teachers
from utils import assert_valid_sequence, log_trainable_parameters, unpack_latents


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
    )

    def main_print(msg):
        if accelerator.is_main_process:
            accelerator.print(msg)

    log_txt_path = os.path.join(args.output_dir, "log.txt")

    def main_write(msg):
        if accelerator.is_main_process:
            with open(log_txt_path, "a", encoding="utf-8") as f:
                f.write(msg)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(args.output_dir, f"log_{stamp}.txt"), "w") as f:
            f.write(f"Program start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\nArguments:\n")
            for key, value in vars(args).items():
                f.write(f"  {key}: {value}\n")
            f.write("=" * 50 + "\n\n")

    device = accelerator.device
    train_dtype = (
        torch.float32
        if args.mixed_precision == "no"
        else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16)
    )
    infer_dtype = torch.bfloat16

    # VAE is only used to decode rollout samples for inspection.
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=infer_dtype
    ).to(device)
    vae.requires_grad_(False)
    vae.enable_tiling()
    image_processor = VaeImageProcessor(16)

    transformer = load_model_with_teachers(args, train_dtype, main_print)

    # Capability-field conditioning: coordinate -> additive embedding.
    task_hook = None
    if args.parsed_cond_vectors is not None:
        task_hook = CapFieldHook(
            transformer,
            cond_vectors=args.parsed_cond_vectors,
            hidden_dim=args.task_cond_hidden_dim,
            main_print=main_print,
        )

    if accelerator.is_main_process:
        log_trainable_parameters(transformer, os.path.join(args.output_dir, "trainable_params.txt"))

    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    main_print(f"--> Optimizing {sum(p.numel() for p in params_to_optimize):,} parameters")

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    ema = None
    if args.ema:
        ema = EMAModuleWrapper(
            params_to_optimize,
            decay=args.ema_decay,
            update_step_interval=args.ema_update_step_interval,
            device=device,
        )
        main_print(f"--> EMA enabled: decay={args.ema_decay}, every {args.ema_update_step_interval} steps")

    # Text encoders stay resident; prompts are encoded lazily with an LRU cache.
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=infer_dtype
    ).to(device)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=infer_dtype
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    for p in text_encoder_2.parameters():
        p.requires_grad_(False)

    encode_cache = {}

    def encode_prompts_on_the_fly(captions):
        pe_list, ppe_list, ti_list = [], [], []
        for cap in captions:
            cached = encode_cache.get(cap)
            if cached is None:
                with torch.no_grad():
                    t5_inputs = tokenizer_2(
                        cap, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
                    )
                    pe = text_encoder_2(t5_inputs.input_ids.to(device))[0].cpu()
                    clip_inputs = tokenizer(
                        cap, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
                    )
                    ppe = text_encoder(clip_inputs.input_ids.to(device)).pooler_output.cpu()
                    ti = torch.zeros(pe.shape[0], 3)
                if len(encode_cache) >= args.encode_cache_size:
                    encode_cache.pop(next(iter(encode_cache)))
                encode_cache[cap] = (pe, ppe, ti)
            else:
                pe, ppe, ti = cached
            pe_list.append(pe)
            ppe_list.append(ppe)
            ti_list.append(ti)
        return (
            torch.cat(pe_list, dim=0),
            torch.cat(ppe_list, dim=0),
            torch.cat(ti_list, dim=0),
        )

    train_dataloader, train_sampler = build_dataloader(args, accelerator)
    transformer, optimizer, train_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader
    )
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    main_print(f"--> Accelerator ready: {accelerator.distributed_type}")
    main_print(
        f"[Data] paired_entries={len(train_dataloader.dataset)} "
        f"gpus={accelerator.num_processes} dl_len={len(train_dataloader)}"
    )

    # sd3_shift: static shift=3.0; flux1_shift: dynamic shift computed from resolution.
    sigma_scheduler = torch.linspace(
        1, 0, args.denoising_steps + 1, dtype=torch.float32, device=device
    )

    def sd3_time_shift(shift, t):
        return (shift * t) / (1 + (shift - 1) * t)

    if args.shift_mode == "flux1_shift":
        image_seq_len = (args.height // 16) * (args.width // 16)
        mu = calculate_shift(image_seq_len)
        sigma_scheduler = sd3_time_shift(math.exp(mu), sigma_scheduler)
        main_print(f"[Scheduler] flux1_shift: image_seq_len={image_seq_len}, mu={mu:.4f}")
    else:
        sigma_scheduler = sd3_time_shift(3.0, sigma_scheduler)
        main_print("[Scheduler] sd3_shift (shift=3.0)")
    guidance = torch.full([1], args.guidance_scale, device=device, dtype=torch.float32)

    assert_valid_sequence(args.train_stepidx, args.denoising_steps)
    main_print(f"--> distill timesteps: {args.train_stepidx}")

    rollout_pools = args.rollout_cond_vectors
    rollout_probs = args.parsed_rollout_probs

    def save_sample_images(all_latents, step):
        try:
            final_latent = all_latents[-1].to(dtype=infer_dtype)
            final_latent = unpack_latents(final_latent, args.height, args.width, 8)
            final_latent = final_latent / vae.config.scaling_factor + vae.config.shift_factor
            with torch.no_grad():
                pixels = vae.decode(final_latent, return_dict=False)[0]
            images = image_processor.postprocess(pixels, output_type="pil")
            for idx, img in enumerate(images):
                img.save(os.path.join(args.output_dir, f"step_{step}_sample{idx + 1}.jpg"))
        except Exception as e:
            main_write(f"Warning: failed to save sample at step {step}: {e}\n")

    global_step = 0
    epoch = 0
    start_time = None
    autocast = accelerator.autocast

    while global_step < args.max_train_steps:
        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}",
        )

        for _, batch in enumerate(progress_bar):
            if global_step >= args.max_train_steps:
                break

            sources = torch.tensor(batch["source"], device=device, dtype=torch.long)

            encoder_hidden_states, pooled_prompt_embeds, text_ids = encode_prompts_on_the_fly(
                batch["captions"]
            )
            encoder_hidden_states = encoder_hidden_states.to(device, dtype=infer_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device, dtype=infer_dtype)
            text_ids = text_ids.to(device, dtype=infer_dtype)

            with accelerator.accumulate(transformer):
                # 1. Rollout: student denoises from noise, conditioned on a per-prompt coordinate.
                with torch.no_grad():
                    all_latents, latent_image_ids = sample_student_trajectory(
                        args,
                        transformer,
                        encoder_hidden_states,
                        pooled_prompt_embeds,
                        text_ids,
                        guidance,
                        sigma_scheduler,
                        autocast,
                        sources,
                        task_hook=task_hook,
                        rollout_pools=rollout_pools,
                        rollout_probs=rollout_probs,
                    )

                # 2. Build the coordinate's teacher target and take the MSE loss.
                loss, per_step_losses = compute_opd_loss(
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
                    step_indices=args.train_stepidx,
                    task_hook=task_hook,
                    rollout_cond=task_hook.last_rollout_omega if task_hook is not None else None,
                    loss_target=args.loss_target,
                    backward_fn=accelerator.backward,
                    ddp_model=transformer,
                )

                # 3. Optimizer step.
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    if accelerator.is_main_process and global_step % args.sample_interval == 0:
                        save_sample_images(all_latents, global_step)

                    per_step_str = " | ".join(
                        f"t{idx}={ls.item():.6f}" for idx, ls in zip(args.train_stepidx, per_step_losses)
                    )
                    log_line = (
                        f"global_step={global_step} loss={loss.item():.6f} [{per_step_str}] "
                        f"grad_norm={grad_norm:.6f} lr={optimizer.param_groups[0]['lr']:.2e}"
                    )
                    if task_hook is not None:
                        log_line += f" e_task_norm={task_hook.last_e_task_norm:.4f}"
                        log_line += " coord=" + " | ".join(
                            "[" + ",".join(f"{v:.2f}" for v in row) + "]"
                            for row in task_hook.last_cond_matrix.detach().float().cpu().tolist()
                        )
                    if start_time is not None:
                        log_line += f" elapsed={time.time() - start_time:.0f}s"
                    main_write(log_line + "\n")
                    start_time = time.time()

                    global_step += 1
                    if ema is not None:
                        ema.step(params_to_optimize, global_step)

                if global_step > 0 and global_step % args.save_interval == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_student_lora(
                            transformer,
                            os.path.join(args.output_dir, f"step_{global_step}"),
                            main_print,
                            task_hook=task_hook,
                        )
                        if ema is not None:
                            ema.copy_ema_to(params_to_optimize, store_temp=True)
                            try:
                                save_student_lora(
                                    transformer,
                                    os.path.join(args.output_dir, f"step_{global_step}_ema"),
                                    main_print,
                                    task_hook=task_hook,
                                )
                            finally:
                                ema.copy_temp_to(params_to_optimize)
                        gc.collect()
                        torch.cuda.empty_cache()

            if global_step >= args.max_train_steps:
                break

        epoch += 1
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()


def _parse_levels(token, where):
    """Parse one coordinate level ("0.7 0.0 1.0") into a float list."""
    parts = str(token).replace(",", " ").split()
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise AssertionError(f"{where}: '{token}' is not a numeric coordinate vector")
    assert values, f"{where}: empty coordinate vector"
    return values


def parse_args():
    p = argparse.ArgumentParser(description="CapField-OPD training")

    # Paths
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument(
        "--source_prompt_txt_paths",
        nargs="+",
        type=str,
        required=True,
        help="One prompt file per capability axis (txt or jsonl). Order defines the axes.",
    )
    p.add_argument("--output_dir", type=str, required=True)

    # Teachers: "LORA_PATH|v1 v2 ... vM" (path 'none' = base model)
    p.add_argument("--experts", nargs="+", type=str, required=True)

    # Coordinates
    p.add_argument(
        "--source_cond_vectors",
        nargs="+",
        type=str,
        default=None,
        help="Per-axis loss coordinate pool, ';'-separated levels, e.g. '1 0; 0.7 0; 1 1'.",
    )
    p.add_argument(
        "--rollout_cond_vectors",
        nargs="+",
        type=str,
        default=None,
        help="Per-axis rollout coordinate pool (';'-separated levels). "
        "Omit to roll out from each axis' own anchor coordinate.",
    )
    p.add_argument(
        "--rollout_cond_probs",
        nargs="+",
        type=str,
        default=None,
        help="Per-axis probabilities for the rollout pool ('none' = uniform).",
    )
    p.add_argument("--task_cond_hidden_dim", type=int, default=256)

    # Training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--encode_cache_size", type=int, default=500)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_interval", type=int, default=200)
    p.add_argument("--sample_interval", type=int, default=5)

    # EMA
    p.add_argument("--ema", action="store_true", default=False)
    p.add_argument("--ema_decay", type=float, default=0.9)
    p.add_argument("--ema_update_step_interval", type=int, default=8)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)

    # Distillation / sampling
    p.add_argument("--denoising_steps", type=int, default=10)
    p.add_argument("--shift_mode", type=str, default="sd3_shift", choices=["sd3_shift", "flux1_shift"])
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--train_stepidx", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--loss_target", type=str, default="velocity",
                   choices=["velocity", "next_latent", "pred_x0"])
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)

    args = p.parse_args()
    return args


def build_teacher_registry(args):
    """Validate --experts and attach the registry / adapter specs to args."""
    num_sources = len(args.source_prompt_txt_paths)

    registry = TeacherRegistry()
    exp_dim = None
    for k, spec in enumerate(args.experts):
        parts = [s.strip() for s in str(spec).split("|", 1)]
        assert len(parts) == 2 and all(parts), f"--experts[{k}] must be 'PATH|v1 v2 ...': '{spec}'"
        path, vec_str = parts
        vec = [float(x) for x in vec_str.replace(",", " ").split()]
        assert vec, f"--experts[{k}] ({path}) is empty"
        if exp_dim is None:
            exp_dim = len(vec)
        else:
            assert len(vec) == exp_dim, f"--experts[{k}] ({path}) vector dim {len(vec)} != {exp_dim}"
        registry.register(path, vec)
    assert exp_dim == num_sources, (
        f"teacher coordinate dim ({exp_dim}) must equal the number of sources ({num_sources})"
    )
    for si in range(num_sources):
        if si not in registry.solo:
            e = [0.0] * num_sources
            e[si] = 1.0
            raise AssertionError(f"missing solo teacher for axis {si}; add an expert with vector {e}")
    args.teacher_registry = registry
    args.teacher_specs = [(entry["name"], entry["path"]) for entry in registry.entries]

    # ---- Loss coordinate pools ----
    needs_vanilla = False
    if args.source_cond_vectors is not None:
        assert len(args.source_cond_vectors) == num_sources, "source_cond_vectors must be per-axis"
        parsed = []
        for si, vec_str in enumerate(args.source_cond_vectors):
            levels = []
            for tok in str(vec_str).split(";"):
                vec = _parse_levels(tok, f"--source_cond_vectors[{si}]")
                for x in vec:
                    assert -1e-6 <= x <= 1.0 + 1e-6, (
                        f"--source_cond_vectors[{si}] level {vec} has value {x} outside [0, 1]"
                    )
                levels.append(vec)
            assert levels, f"--source_cond_vectors[{si}] parsed no levels"
            parsed.append(levels)
        dims = {len(lv) for lv_all in parsed for lv in lv_all}
        assert len(dims) == 1, f"all coordinate levels must share one dim, got {sorted(dims)}"
        cond_dim = dims.pop()
        assert cond_dim == num_sources, (
            f"coordinate dim ({cond_dim}) must equal the number of sources ({num_sources})"
        )
        for lv_all in parsed:
            for lv in lv_all:
                mix = registry.decompose(lv)
                if any(name == "vanilla" for name, _ in mix):
                    needs_vanilla = True
        args.parsed_cond_vectors = parsed
    else:
        args.parsed_cond_vectors = None

    # ---- Rollout coordinate pools + sampling probabilities ----
    args.parsed_rollout_probs = None
    if args.rollout_cond_vectors:
        assert args.parsed_cond_vectors is not None, (
            "--rollout_cond_vectors requires --source_cond_vectors"
        )
        cond_dim = len(args.parsed_cond_vectors[0][0])
        assert len(args.rollout_cond_vectors) == num_sources, "rollout pools must be per-axis"
        raw_pools = []
        for s, entry in enumerate(args.rollout_cond_vectors):
            pool = []
            for tok in str(entry).split(";"):
                vec = _parse_levels(tok, f"--rollout_cond_vectors[{s}]")
                assert len(vec) == cond_dim, f"--rollout_cond_vectors[{s}] level {vec} dim != {cond_dim}"
                pool.append(vec)
            assert pool, f"--rollout_cond_vectors[{s}] has no valid level"
            raw_pools.append(pool)

        parsed_probs = [None] * num_sources
        has_prob = False
        if args.rollout_cond_probs is not None:
            assert len(args.rollout_cond_probs) == num_sources, "rollout probs must be per-axis"
            for si, w_str in enumerate(args.rollout_cond_probs):
                ws = str(w_str).strip()
                if not ws or ws.lower() == "none":
                    continue
                pw = [float(x) for x in ws.replace(",", " ").split()]
                assert pw and all(x > 0 for x in pw), f"--rollout_cond_probs[{si}] invalid"
                assert len(pw) == len(raw_pools[si]), f"--rollout_cond_probs[{si}] count != pool size"
                total = sum(pw)
                parsed_probs[si] = [x / total for x in pw]
                has_prob = True

        # Every rollout level enters the teacher target path, so it must be decomposable.
        for pool in raw_pools:
            for lv in pool:
                try:
                    registry.decompose(lv)
                except AssertionError as err:
                    raise AssertionError(f"[Rollout] level {lv}: {err}") from None
                if max(lv) < 1.0 - 1e-6 or sum(1 for x in lv if abs(x) > 1e-6) == 2 or not any(lv):
                    needs_vanilla = True

        args.rollout_cond_vectors = raw_pools if any(raw_pools) else None
        args.parsed_rollout_probs = parsed_probs if has_prob else None

    args.needs_vanilla = needs_vanilla


if __name__ == "__main__":
    parsed_args = parse_args()
    build_teacher_registry(parsed_args)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[Teachers] {len(parsed_args.teacher_registry.entries)} adapter(s):")
        print(parsed_args.teacher_registry.describe())
    try:
        main(parsed_args)
    except Exception:
        import traceback

        os.makedirs(parsed_args.output_dir, exist_ok=True)
        with open(os.path.join(parsed_args.output_dir, "error_traceback.txt"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise
