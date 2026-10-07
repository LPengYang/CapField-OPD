"""CapField-OPD inference: one prompt, several coordinates, one image each.

The student checkpoint is a folder that holds three files:
    adapter_config.json
    adapter_model.safetensors
    task_conditioner.pt        (the capability-coordinate conditioner)

All coordinates in one run share the same noise, so the images differ only by
the capability-field coordinate.

Examples:
    # Released checkpoint, downloaded straight from the Hub (see README):
    python inference.py \
        --ckpt Yang18/CapField-OPD --ckpt_subfolder student_ckpt \
        --prompt "A close-up of a sign that reads 'HELLO WORLD'" \
        --coords "1.0 0.0 0.0" "0.0 0.0 1.0" "1.0 0.0 1.0"

    # Local checkpoint folder:
    python inference.py \
        --ckpt runs/mopd/step_2000 \
        --prompt "A close-up of a sign that reads 'HELLO WORLD'" \
        --coords "1.0 0.0 0.0" "0.0 1.0 0.0"
"""

import argparse
import math
import os

import torch
from diffusers import AutoencoderKL, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from peft import PeftModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from modeling import CONDITIONER_FILENAMES, CapFieldHook, find_conditioner_file
from utils import pack_latents, prepare_latent_image_ids, unpack_latents


def parse_args():
    p = argparse.ArgumentParser(description="CapField-OPD inference")
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Student checkpoint: a local folder, or a Hugging Face repo id "
        "(e.g. Yang18/CapField-OPD). If it is a repo id, the checkpoint is "
        "downloaded automatically (see --ckpt_subfolder).",
    )
    p.add_argument(
        "--ckpt_subfolder",
        type=str,
        default=None,
        help="Subfolder that holds the checkpoint inside --ckpt, e.g. "
        "student_ckpt (Hub) or step_N (training output). Used both for a local "
        "dir and for a Hugging Face repo id.",
    )
    p.add_argument(
        "--base_model",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="FLUX base model: a local dir or a Hugging Face repo id.",
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument(
        "--coords",
        nargs="+",
        type=str,
        default=None,
        help="One or more coordinates, e.g. --coords '1.0 0.0 0.0' '0.0 0.0 1.0' "
        "(3-axis [geneval, ocr, aesthetics] for the released student). Defaults "
        "to a small demo sweep over the first two axes.",
    )
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--denoising_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--shift_mode", type=str, default="flux1_shift",
                   choices=["sd3_shift", "flux1_shift"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs/inference")
    return p.parse_args()


def resolve_source_dir(path, subfolder=None):
    """Return a local directory for `path`.

    If `path` is an existing local directory it is used as-is; when `subfolder`
    is given it is appended only if that subdirectory actually exists. Otherwise
    `path` is treated as a Hugging Face repo id and downloaded (only `subfolder`
    when given).
    """
    if os.path.isdir(path):
        if subfolder:
            candidate = os.path.join(path, subfolder)
            if os.path.isdir(candidate):
                return candidate
        return path

    from huggingface_hub import snapshot_download

    kwargs = {}
    if subfolder:
        kwargs["allow_patterns"] = [f"{subfolder}/*", f"{subfolder}/**"]
    local = snapshot_download(repo_id=path, **kwargs)
    return os.path.join(local, subfolder) if subfolder else local


def load_student(ckpt_dir, base_model_path, device):
    transformer = FluxTransformer2DModel.from_pretrained(
        base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.requires_grad_(False)

    model = PeftModel.from_pretrained(
        transformer, ckpt_dir, torch_dtype=torch.bfloat16, adapter_name="student"
    )
    model.set_adapter("student")
    model.eval()

    cond_path = find_conditioner_file(ckpt_dir)
    assert cond_path is not None, (
        f"no capability-coordinate conditioner in {ckpt_dir}: expected one of "
        f"{list(CONDITIONER_FILENAMES)}"
    )
    meta = torch.load(cond_path, map_location="cpu")["meta"]
    hook = CapFieldHook(
        model, cond_vectors=meta["cond_vectors"], hidden_dim=meta["hidden_dim"]
    )
    assert hook.load(ckpt_dir), f"failed to load {cond_path}"
    hook.conditioner.to(device=device, dtype=torch.bfloat16)
    return model, hook, meta


@torch.no_grad()
def encode_prompt(prompt, base_model_path, device):
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        base_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device).eval()
    text_encoder_2 = T5EncoderModel.from_pretrained(
        base_model_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    ).to(device).eval()

    t5_out = tokenizer_2(prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    encoder_hidden_states = text_encoder_2(t5_out.input_ids.to(device))[0]

    clip_out = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    pooled = text_encoder(clip_out.input_ids.to(device)).pooler_output

    text_ids = torch.zeros(1, 3, device=device)
    return encoder_hidden_states, pooled, text_ids


@torch.no_grad()
def generate(model, hook, coords, encoder_hidden_states, pooled, text_ids, args, device):
    batch = len(coords)

    # One shared noise for every coordinate.
    noise = torch.randn(
        (1, 16, args.height // 8, args.width // 8), device=device, dtype=torch.bfloat16
    ).repeat(batch, 1, 1, 1)
    latents = pack_latents(noise, batch, 16, args.height // 8, args.width // 8)
    img_ids = prepare_latent_image_ids(batch, args.height // 16, args.width // 16, device, torch.bfloat16)
    txt_ids = text_ids.expand(encoder_hidden_states.shape[1], -1)

    sigmas = torch.linspace(1, 0, args.denoising_steps + 1, device=device)
    if args.shift_mode == "flux1_shift":
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift

        image_seq_len = (args.height // 16) * (args.width // 16)
        eff = math.exp(calculate_shift(image_seq_len))
        sigmas = (eff * sigmas) / (1 + (eff - 1) * sigmas)
    else:
        sigmas = (3.0 * sigmas) / (1 + (3.0 - 1) * sigmas)
    guidance = torch.full([1], args.guidance_scale, device=device)

    hook.set_coord(torch.tensor(coords, device=device))
    hook.enable()
    for i in range(args.denoising_steps):
        t = sigmas[i].expand(batch)
        with torch.autocast("cuda", torch.bfloat16):
            velocity = model(
                hidden_states=latents,
                timestep=t,
                guidance=guidance.expand(batch),
                pooled_projections=pooled.repeat(batch, 1),
                encoder_hidden_states=encoder_hidden_states.repeat(batch, 1, 1),
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
        latents = (latents.float() - (sigmas[i] - sigmas[i + 1]) * velocity.float()).to(torch.bfloat16)
    hook.disable()
    return latents


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    ckpt_dir = resolve_source_dir(args.ckpt, args.ckpt_subfolder)
    base_dir = resolve_source_dir(args.base_model)

    model, hook, meta = load_student(ckpt_dir, base_dir, device)
    print(f"[ckpt] {ckpt_dir}")
    print(f"[base] {base_dir}")
    print(f"[meta] coord_dim={meta['cond_dim']} hidden={meta['hidden_dim']}")

    if args.coords is not None:
        coords = [[float(x) for x in c.replace(",", " ").split()] for c in args.coords]
    else:
        dim = meta["cond_dim"]
        coords = []
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            c = [0.0] * dim
            c[0] = 1.0 - w
            if dim > 1:
                c[1] = w
            coords.append(c)
    for c in coords:
        assert len(c) == meta["cond_dim"], f"coordinate {c} must have dim {meta['cond_dim']}"
    print(f"[coords] {coords}")

    encoder_hidden_states, pooled, text_ids = encode_prompt(args.prompt, base_dir, device)
    final_latents = generate(model, hook, coords, encoder_hidden_states, pooled, text_ids, args, device)

    vae = AutoencoderKL.from_pretrained(
        base_dir, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device)
    image_processor = VaeImageProcessor(16)

    latents_2d = unpack_latents(final_latents, args.height, args.width, 8)
    latents_2d = latents_2d / vae.config.scaling_factor + vae.config.shift_factor
    images = image_processor.postprocess(vae.decode(latents_2d, return_dict=False)[0], output_type="pil")

    for coord, img in zip(coords, images):
        tag = "_".join(f"{x:.2f}" for x in coord)
        path = os.path.join(args.output_dir, f"coord_{tag}.jpg")
        img.save(path)
        print(f"[saved] {path} (coord={coord})")


if __name__ == "__main__":
    main()
