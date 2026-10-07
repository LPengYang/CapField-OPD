"""GenEval reward (optional, used by reward_server.py).

Ported from the DiffusionOPD GenEval scorer. Detects objects with a Mask2Former
COCO detector and classifies colors with CLIP ViT-L-14.
Requires: mmdet (2.x) + mmcv (1.x) + open-clip-torch + clip-benchmark.

Only imported when the `geneval` reward is requested.
"""

import json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageOps

warnings.filterwarnings("ignore")

import mmdet
from mmdet.apis import inference_detector, init_detector

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc

zsc.tqdm = lambda it, *args, **kwargs: it

DEFAULT_CKPT_DIR = os.environ.get("CAPFIELD_GENEVAL_CKPT_DIR", "reward_ckpts")
DEFAULT_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geneval_assets")
DEFAULT_CLIP_PRETRAINED = "hf-hub:timm/ViT-L-14"
OBJECT_DETECTOR = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99"


def load_geneval(DEVICE, ckpt_dir=None, clip_pretrained=None, assets_dir=None):
    def timed(fn):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            print(f"{fn.__name__!r} executed in {time.time() - start:.3f}s", file=sys.stderr)
            return result

        return wrapper

    @timed
    def load_models():
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(mmdet.__file__)),
            "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
        )
        ckpt_path = os.path.join(ckpt_dir or DEFAULT_CKPT_DIR, f"{OBJECT_DETECTOR}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"GenEval detector checkpoint not found at {ckpt_path}. Download it with:\n"
                "wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/"
                "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/"
                "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
            )
        object_detector = init_detector(config_path, ckpt_path, device=DEVICE)

        clip_spec = clip_pretrained or DEFAULT_CLIP_PRETRAINED
        if os.path.isdir(clip_spec) or os.path.isfile(clip_spec):
            clip_model, _, transform = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained=clip_spec, device=DEVICE
            )
            tokenizer = open_clip.get_tokenizer("ViT-L-14")
        else:
            clip_model, _, transform = open_clip.create_model_and_transforms(clip_spec, device=DEVICE)
            tokenizer = open_clip.get_tokenizer(clip_spec)

        obj_names = os.path.join(assets_dir or DEFAULT_ASSETS_DIR, "object_names.txt")
        with open(obj_names) as cls_file:
            classnames = [line.strip() for line in cls_file]
        return object_detector, (clip_model, transform, tokenizer), classnames

    COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
    COLOR_CLASSIFIERS = {}

    class ImageCrops(torch.utils.data.Dataset):
        def __init__(self, image: Image.Image, objects):
            self._image = image.convert("RGB")
            self._blank = Image.new("RGB", image.size, color="#999")
            self._objects = objects

        def __len__(self):
            return len(self._objects)

        def __getitem__(self, index):
            box, mask = self._objects[index]
            if mask is not None:
                image = Image.composite(self._image, self._blank, Image.fromarray(mask))
            else:
                image = self._image
            return (transform(image.crop(box[:4])), 0)

    def color_classification(image, bboxes, classname):
        if classname not in COLOR_CLASSIFIERS:
            COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                clip_model,
                tokenizer,
                COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object",
                ],
                str(DEVICE),
            )
        clf = COLOR_CLASSIFIERS[classname]
        dataloader = torch.utils.data.DataLoader(ImageCrops(image, bboxes), batch_size=16, num_workers=0)
        with torch.no_grad():
            pred, _ = zsc.run_classification(clip_model, clf, dataloader, str(DEVICE))
            return [COLORS[index.item()] for index in pred.argmax(1)]

    def compute_iou(box_a, box_b):
        area = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
        inter = area(
            [
                max(box_a[0], box_b[0]),
                max(box_a[1], box_b[1]),
                min(box_a[2], box_b[2]),
                min(box_a[3], box_b[3]),
            ]
        )
        union = area(box_a) + area(box_b) - inter
        return inter / union if union else 0

    def relative_position(obj_a, obj_b):
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised) < 1e-3):
            return set()
        dx, dy = revised / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations

    def evaluate(image, objects, metadata):
        correct = True
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[: req["count"]]
            if len(found_objects) < req["count"]:
                correct = matched = False
            else:
                if "color" in req:
                    colors = color_classification(image, found_objects, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                if expected_rel not in relative_position(obj, target_obj):
                                    correct = matched = False
                                    break
                            if not matched:
                                break
            matched_groups.append(found_objects if matched else None)
        for req in metadata.get("exclude", []):
            if len(objects.get(req["class"], [])) >= req["count"]:
                correct = False
        return correct, ""

    def evaluate_reward(image, objects, metadata):
        correct = True
        rewards = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])
            count_norm = req["count"] if req["count"] != 0 else 1
            rewards.append(1 - abs(req["count"] - len(found_objects)) / count_norm)
            if len(found_objects) != req["count"]:
                correct = matched = False
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    colors = color_classification(image, found_objects, classname)
                    rewards.append(1 - abs(req["count"] - colors.count(req["color"])) / count_norm)
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        rewards.append(0.0)
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                if expected_rel not in relative_position(obj, target_obj):
                                    correct = matched = False
                                    rewards.append(0.0)
                                    break
                            if not matched:
                                break
                        rewards.append(1.0)
            matched_groups.append(found_objects if matched else None)
        reward = sum(rewards) / len(rewards) if rewards else 0
        return correct, reward, ""

    def evaluate_image(image_pils, metadatas, only_strict):
        # mmdet expects BGR input.
        results = inference_detector(
            object_detector, [np.array(img)[:, :, ::-1] for img in image_pils]
        )
        ret = []
        for result, image_pil, metadata in zip(results, image_pils, metadatas):
            bbox = result[0] if isinstance(result, tuple) else result
            segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            image = ImageOps.exif_transpose(image_pil)
            detected = {}
            confidence_threshold = THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
            for index, classname in enumerate(classnames):
                ordering = np.argsort(bbox[index][:, 4])[::-1]
                ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
                ordering = ordering[:MAX_OBJECTS].tolist()
                detected[classname] = []
                while ordering:
                    max_obj = ordering.pop(0)
                    detected[classname].append(
                        (bbox[index][max_obj], None if segm is None else segm[index][max_obj])
                    )
                    ordering = [
                        obj
                        for obj in ordering
                        if NMS_THRESHOLD == 1
                        or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
                    ]
                if not detected[classname]:
                    del detected[classname]
            is_strict_correct, score, reason = evaluate_reward(image, detected, metadata)
            is_correct = False if only_strict else evaluate(image, detected, metadata)[0]
            ret.append(
                {
                    "tag": metadata["tag"],
                    "prompt": metadata["prompt"],
                    "correct": is_correct,
                    "strict_correct": is_strict_correct,
                    "score": score,
                    "reason": reason,
                }
            )
        return ret

    object_detector, (clip_model, transform, tokenizer), classnames = load_models()
    THRESHOLD = 0.3
    COUNTING_THRESHOLD = 0.9
    MAX_OBJECTS = 16
    NMS_THRESHOLD = 1.0
    POSITION_THRESHOLD = 0.1

    @torch.no_grad()
    def compute_geneval(images, metadatas, only_strict=False):
        required_keys = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
        scores, strict_rewards, rewards = [], [], []
        grouped_strict = defaultdict(list)
        grouped = defaultdict(list)
        normalized = []
        for image, metadata in zip(images, metadatas):
            if not isinstance(metadata, dict):
                metadata = {}
            metadata = dict(metadata)
            metadata.setdefault("tag", "single_object")
            metadata.setdefault("include", [])
            metadata.setdefault("prompt", "")
            normalized.append(metadata)
        for result in evaluate_image(images, normalized, only_strict=only_strict):
            strict_rewards.append(1.0 if result["strict_correct"] else 0.0)
            scores.append(result["score"])
            rewards.append(1.0 if result["correct"] else 0.0)
            tag = result["tag"]
            for key in required_keys:
                if key != tag:
                    grouped_strict[key].append(-10.0)
                    grouped[key].append(-10.0)
                else:
                    grouped_strict[tag].append(1.0 if result["strict_correct"] else 0.0)
                    grouped[tag].append(1.0 if result["correct"] else 0.0)
        return scores, rewards, strict_rewards, dict(grouped), dict(grouped_strict)

    return compute_geneval


if __name__ == "__main__":
    data = {
        "images": [Image.open("test.png")],
        "metadatas": [
            {
                "tag": "color_attr",
                "include": [
                    {"class": "giraffe", "count": 1, "color": "red"},
                    {"class": "stop sign", "count": 1, "color": "white"},
                ],
                "prompt": "a photo of a brown giraffe and a white stop sign",
            }
        ],
        "only_strict": False,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    compute = load_geneval(device)
    print(compute(**data)[:3])
