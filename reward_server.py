"""Optional reward server for evaluation (not used by training).

Exposes an HTTP API so evaluation scripts can score generated images. Supports:
    hpsv3 | clip | pickscore | paddleocr | geneval

Run it in the reward environment, one process per GPU, e.g.:
    python reward_server.py --reward_name paddleocr hpsv3 --port 15000 --gpu 0 \
        --paddleocr_model_dir /path/to/paddleocr/whl

The heavy model paths are supplied via CLI arguments so nothing is hard-coded.
"""

import argparse
import base64
import io
import os
from typing import List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

app = FastAPI()
reward_models = {}
DEVICE = None
_geneval_fn = None
_geneval_only_strict = False


class RewardRequest(BaseModel):
    images_b64: List[str]
    captions: List[str]
    reward_name: str
    metadatas: Optional[List[dict]] = None


class RewardResponse(BaseModel):
    scores: List[float]


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _per_image_captions(captions, num_images):
    return captions if len(captions) == num_images else [captions[0]] * num_images


# --------------------------------------------------------------------------- #
# Reward functions
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_reward_hpsv3(images_list, captions, metadatas=None):
    captions = _per_image_captions(captions, len(images_list))
    with torch.amp.autocast("cuda"):
        scores = reward_models["hpsv3"].reward(images_list, captions)
    return scores[:, 0].float().cpu().tolist()


@torch.no_grad()
def compute_reward_clip(images_list, captions, metadatas=None):
    import torch.nn.functional as F

    rm = reward_models["clip"]
    captions = _per_image_captions(captions, len(images_list))
    scores = []
    with torch.amp.autocast("cuda"):
        for image_pil, caption in zip(images_list, captions):
            image = rm["preprocess_val"](image_pil).unsqueeze(0).to(DEVICE)
            text = rm["tokenizer"](caption).to(DEVICE)
            image_features = F.normalize(rm["model"].encode_image(image), dim=-1)
            text_features = F.normalize(rm["model"].encode_text(text), dim=-1)
            scores.append((image_features @ text_features.T)[0].float())
    return torch.stack(scores).flatten().cpu().tolist()


@torch.no_grad()
def compute_reward_pickscore(images_list, captions, metadatas=None):
    rm = reward_models["pickscore"]
    captions = _per_image_captions(captions, len(images_list))
    with torch.amp.autocast("cuda"):
        image_inputs = rm["processor"](
            images=images_list, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(DEVICE)
        text_inputs = rm["processor"](
            text=captions, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(DEVICE)
        image_embs = rm["model"].get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
        text_embs = rm["model"].get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        scores = rm["model"].logit_scale.exp() * (text_embs @ image_embs.T).diagonal()
    return scores.float().cpu().tolist()


@torch.no_grad()
def compute_reward_paddleocr(images_list, captions, metadatas=None):
    from Levenshtein import distance

    captions = _per_image_captions(captions, len(images_list))
    ocr = reward_models["paddleocr"]
    scores = []
    for image_pil, prompt in zip(images_list, captions):
        target = prompt.split('"')[1] if '"' in prompt else prompt
        img_np = np.array(image_pil.convert("RGB"))
        try:
            result = ocr.ocr(img_np, cls=False)
        except Exception as e:
            print(f"[OCR] inference failed: {e}")
            scores.append(0.0)
            continue
        recognized = ""
        if result and result[0]:
            recognized = "".join(res[1][0] for res in result[0] if res[1][1] > 0)
        recognized = recognized.replace(" ", "").lower()
        target_clean = target.replace(" ", "").lower()
        dist = min(distance(recognized, target_clean), len(target_clean))
        scores.append(1.0 - dist / max(len(target_clean), 1))
    return scores


def compute_reward_geneval(images_list, captions, metadatas=None):
    if _geneval_fn is None:
        raise RuntimeError("geneval model not loaded")
    if not metadatas:
        metadatas = [{"tag": "single_object", "include": [], "prompt": captions[0]} for _ in images_list]
    elif len(metadatas) != len(images_list):
        metadatas = [metadatas[0]] * len(images_list)
    out = []
    for start in range(0, len(images_list), 64):
        scores, _, strict_rewards, _, _ = _geneval_fn(
            images_list[start:start + 64],
            metadatas[start:start + 64],
            only_strict=_geneval_only_strict,
        )
        out.extend(float(r) for r in (strict_rewards if _geneval_only_strict else scores))
    return out


REWARD_FN_MAP = {
    "hpsv3": compute_reward_hpsv3,
    "clip": compute_reward_clip,
    "pickscore": compute_reward_pickscore,
    "paddleocr": compute_reward_paddleocr,
    "geneval": compute_reward_geneval,
}


@app.post("/compute_reward", response_model=RewardResponse)
async def compute_reward(req: RewardRequest):
    images = [b64_to_pil(b) for b in req.images_b64]
    scores = REWARD_FN_MAP[req.reward_name](images, req.captions, metadatas=req.metadatas)
    return RewardResponse(scores=scores)


@app.get("/health")
async def health():
    return {"status": "ok", "loaded_models": list(reward_models.keys()), "device": str(DEVICE)}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def init_models(args, device):
    global DEVICE, _geneval_fn, _geneval_only_strict
    DEVICE = device

    if "hpsv3" in args.reward_name:
        from hpsv3_reward import HPSv3RewardInferencer

        print(f"[GPU {device}] loading HPSv3 ...")
        reward_models["hpsv3"] = HPSv3RewardInferencer(
            config_path=args.hpsv3_config, checkpoint_path=args.hpsv3_ckpt, device=device
        )

    if "clip" in args.reward_name:
        from open_clip import create_model_from_pretrained, get_tokenizer

        print(f"[GPU {device}] loading CLIP from {args.clip_model_path} ...")
        model, preprocess = create_model_from_pretrained(f"local-dir:{args.clip_model_path}")
        reward_models["clip"] = {
            "model": model.to(device).eval(),
            "preprocess_val": preprocess,
            "tokenizer": get_tokenizer("ViT-H-14"),
        }

    if "pickscore" in args.reward_name:
        from transformers import AutoModel, AutoProcessor

        print(f"[GPU {device}] loading PickScore from {args.pickscore_model_path} ...")
        reward_models["pickscore"] = {
            "processor": AutoProcessor.from_pretrained(args.pickscore_processor_path),
            "model": AutoModel.from_pretrained(args.pickscore_model_path).to(device).eval(),
        }

    if "paddleocr" in args.reward_name:
        print(f"[GPU {device}] loading PaddleOCR (CPU, lang={args.ocr_lang}) ...")
        reward_models["paddleocr"] = _init_paddleocr(args.paddleocr_model_dir, args.ocr_lang)

    if "geneval" in args.reward_name:
        from geneval_reward import load_geneval

        print(f"[GPU {device}] loading GenEval ...")
        _geneval_fn = load_geneval(
            device,
            ckpt_dir=args.geneval_ckpt_dir,
            clip_pretrained=args.geneval_clip_path,
            assets_dir=args.geneval_assets_dir,
        )
        _geneval_only_strict = bool(args.geneval_strict)
        reward_models["geneval"] = _geneval_fn


def _init_paddleocr(model_base, lang="en"):
    cpu = os.cpu_count() or 64
    procs = max(1, int(os.environ.get("PROC_PER_NODE", 8)))
    threads = max(1, cpu // procs // 2)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))

    from paddleocr import PaddleOCR

    class PaddleOCRWrapper:
        def __init__(self):
            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang=lang,
                det_model_dir=os.path.join(model_base, "det/en/en_PP-OCRv3_det_infer"),
                rec_model_dir=os.path.join(model_base, "rec/en/en_PP-OCRv4_rec_infer"),
                cls_model_dir=os.path.join(model_base, "cls/ch_ppocr_mobile_v2.0_cls_infer"),
                use_gpu=False,
                show_log=False,
            )

        def ocr(self, img, cls=False):
            return self._ocr.ocr(img, cls=cls)

    return PaddleOCRWrapper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_name", nargs="+", type=str, default=["paddleocr"],
                        choices=["hpsv3", "clip", "pickscore", "paddleocr", "geneval"])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--ocr_lang", type=str, default="en", choices=["en", "ch", "japan", "korean"])
    parser.add_argument("--paddleocr_model_dir", type=str, default=os.environ.get("CAPFIELD_PADDLEOCR_DIR", "path/to/paddleocr/whl"))
    parser.add_argument("--hpsv3_config", type=str, default=None)
    parser.add_argument("--hpsv3_ckpt", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--pickscore_model_path", type=str, default=None)
    parser.add_argument("--pickscore_processor_path", type=str, default=None)
    parser.add_argument("--geneval_ckpt_dir", type=str, default=None)
    parser.add_argument("--geneval_clip_path", type=str, default=None)
    parser.add_argument("--geneval_assets_dir", type=str, default=None)
    parser.add_argument("--geneval_strict", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    init_models(args, device)

    print(f"[GPU {args.gpu}] starting reward server on port {args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")
