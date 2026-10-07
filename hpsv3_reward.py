"""HPSv3 reward model wrapper (optional, used by reward_server.py).

Requires the `hpsv3` package to be importable. Paths default to the values
below and can be overridden through `reward_server.py` arguments.
"""

import os

import torch
from PIL import Image


DEFAULT_HPSV3_CONFIG = os.environ.get(
    "CAPFIELD_HPSV3_CONFIG", "path/to/HPSv3_7B.yaml"
)
DEFAULT_HPSV3_CKPT = os.environ.get(
    "CAPFIELD_HPSV3_CKPT", "path/to/HPSv3.safetensors"
)


class HPSv3RewardInferencer:
    def __init__(self, config_path=None, checkpoint_path=None, device="cuda", differentiable=False):
        from hpsv3.dataset.data_collator_qwen import (
            INSTRUCTION,
            prompt_with_special_token,
            prompt_without_special_token,
        )
        from hpsv3.dataset.utils import process_vision_info
        from hpsv3.train import create_model_and_processor
        from hpsv3.utils.parser import (
            DataConfig,
            ModelConfig,
            PEFTLoraConfig,
            TrainingConfig,
            parse_args_with_yaml,
        )

        self._instruction = INSTRUCTION
        self._prompt_with_special_token = prompt_with_special_token
        self._prompt_without_special_token = prompt_without_special_token
        self._process_vision_info = process_vision_info

        config_path = config_path or DEFAULT_HPSV3_CONFIG
        checkpoint_path = checkpoint_path or DEFAULT_HPSV3_CKPT
        (data_config, training_args, model_config, peft_lora_config), config_path = parse_args_with_yaml(
            (DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig), config_path, is_train=False
        )
        training_args.output_dir = os.path.join(
            training_args.output_dir, os.path.basename(config_path).split(".")[0]
        )
        model, processor, _ = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
            differentiable=differentiable,
        )

        self.device = device
        self.use_special_tokens = model_config.use_special_tokens

        if checkpoint_path.endswith(".safetensors"):
            import safetensors.torch

            state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]

        has_language_model = any("model.language_model" in k for k in model.state_dict().keys())
        lacks_language_model = not any("language_model" in k for k in state_dict.keys())
        if has_language_model and lacks_language_model:
            updated = {}
            for key, value in state_dict.items():
                if "visual" in key:
                    updated[key.replace("visual", "model.visual")] = value
                elif "model" in key:
                    updated[key.replace("model", "model.language_model")] = value
                else:
                    updated[key] = value
            state_dict = updated

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model
        self.processor = processor
        self.model.to(self.device)
        self.data_config = data_config

    @torch.inference_mode()
    def reward(self, image_paths, prompts):
        max_pixels = 256 * 28 * 28
        message_list = []
        for text, image in zip(prompts, image_paths):
            message_list.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image, "min_pixels": max_pixels, "max_pixels": max_pixels},
                            {
                                "type": "text",
                                "text": (
                                    self._instruction.format(text_prompt=text) + self._prompt_with_special_token
                                    if self.use_special_tokens
                                    else self._prompt_without_special_token
                                ),
                            },
                        ],
                    }
                ]
            )
        image_inputs, _ = self._process_vision_info(message_list)
        batch = self.processor(
            text=self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        return self.model(return_dict=True, **batch)["logits"]


if __name__ == "__main__":
    inferencer = HPSv3RewardInferencer(device="cuda")
    caption = "A beautiful woman is drinking on the street"
    images = [Image.new("RGB", (512, 512), "white")]
    print(inferencer.reward(images, [caption])[:, 0])
