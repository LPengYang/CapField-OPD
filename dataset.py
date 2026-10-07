"""Multi-source prompt dataset.

Every source (a capability axis) owns exactly one prompt file, either:
  - a `.txt` file with one prompt per line, or
  - a `.jsonl` file where each line is a dict with a "prompt" field.

Each item pairs one prompt per source. The collate function flattens it into
`num_sources * batch_size` samples carrying a `source` index, so every step
sees one prompt from every capability axis.

Text is encoded lazily at train time; nothing is pre-encoded here.
"""

import json

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class MultiSourcePromptDataset(Dataset):
    def __init__(self, prompt_paths, base_seed=0):
        """`prompt_paths[i]` is the prompt file for source (capability axis) i."""
        self.captions = []  # captions[src] = list[str]
        self._indices = []  # deterministic per-source shuffle

        for si, path in enumerate(prompt_paths):
            path = str(path).strip()
            assert path, f"source {si}: empty prompt path"
            if path.endswith(".jsonl"):
                caps = []
                with open(path, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        assert "prompt" in item, f"{path}:{line_no}: jsonl line has no 'prompt'"
                        caps.append(item["prompt"])
                print(f"[Data] source {si}: {len(caps)} prompts (jsonl) from {path}")
            else:
                with open(path, "r", encoding="utf-8") as f:
                    caps = [line.strip() for line in f if line.strip()]
                print(f"[Data] source {si}: {len(caps)} prompts (txt) from {path}")
            assert caps, f"source {si}: empty prompt file {path}"
            self.captions.append(caps)
            g = torch.Generator().manual_seed(base_seed + si)
            self._indices.append(torch.randperm(len(caps), generator=g).tolist())

        # Indexable length is limited by the smallest source.
        self.length = min(len(c) for c in self.captions)
        assert self.length > 0, "all sources are empty"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {str(si): self.captions[si][self._indices[si][idx]]
                for si in range(len(self.captions))}


def collate_fn(batch):
    """Return flat `captions` / `source` lists of length S*B."""
    num_sources = len(batch[0])
    captions, source = [], []
    for src_idx in range(num_sources):
        for b in batch:
            captions.append(b[str(src_idx)])
        source.extend([src_idx] * len(batch))
    return {"captions": captions, "source": source}


def build_dataloader(args, accelerator):
    dataset = MultiSourcePromptDataset(args.source_prompt_txt_paths, base_seed=args.seed)
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    return dataloader, sampler
