"""Model, teacher registry and capability-field conditioning.

A capability axis is one skill (M axes in total); a coordinate is a length-M
weight vector over the axes. The teacher target velocity for a coordinate is
built by inclusion-exclusion over the registered teachers:

    single axis j, strength a          : v = (1-a)·v_base + a·v_j
    two axes i,j, strengths a,b, m=min : v = (1-a-b+m)·v_base
                                           + (a-m)·v_i + (b-m)·v_j + m·v_ij

`v_base` is the raw pretrained model (zero LoRA). Teacher targets only support
up to two active axes; the student accepts any coordinate at inference.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FluxTransformer2DModel
from peft import LoraConfig, get_peft_model

from utils import unwrap

_DECOMP_EPS = 1e-6

# Capability-coordinate conditioner file inside a student checkpoint.
# We save it as "task_conditioner.pt" to match the reference trainer
# (multirereward/train_multi_teacher_opd) and the released Hub checkpoints.
# Legacy CapField-OPD checkpoints used "cap_field_conditioner.pt"; both names
# are accepted when loading.
CONDITIONER_FILENAMES = ("task_conditioner.pt", "cap_field_conditioner.pt")


def find_conditioner_file(directory):
    """Return the path of the conditioner file inside `directory`, or None."""
    for name in CONDITIONER_FILENAMES:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


LORA_TARGET_MODULES = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]


class TeacherRegistry:
    """Maps a coordinate vector to a weighted mixture of teacher adapters."""

    def __init__(self):
        self.entries = []          # [{"name", "path", "path_norm", "vec"}]
        self._by_path = {}         # normalized path -> entry (each LoRA loaded once)
        self.solo = {}             # axis index -> entry (single-axis teacher)
        self.joint = {}            # frozenset({i, j}) -> entry (two-axis teacher)

    def register(self, path, vec):
        """Register one teacher; returns its adapter name ("teacher_N")."""
        key = os.path.normpath(str(path))
        norm = tuple(round(float(v), 6) for v in vec)
        existing = self._by_path.get(key)
        if existing is not None:
            same = len(existing["vec"]) == len(norm) and all(
                abs(a - b) <= 1e-6 for a, b in zip(existing["vec"], norm)
            )
            assert same, (
                f"teacher path '{path}' registered twice with different vectors "
                f"({list(existing['vec'])} vs {list(norm)})"
            )
            return existing["name"]

        support = tuple(i for i, v in enumerate(norm) if abs(v) > _DECOMP_EPS)
        assert support, f"teacher '{path}' has an all-zero coordinate vector: {list(norm)}"
        entry = {
            "name": f"teacher_{len(self.entries)}",
            "path": str(path),
            "path_norm": key,
            "vec": norm,
        }
        if len(support) == 1:
            prev = self.solo.get(support[0])
            assert prev is None, (
                f"axis {support[0]} already has a solo teacher ('{prev['path']}'); "
                f"only one solo teacher per axis is allowed"
            )
            self.solo[support[0]] = entry
        elif len(support) == 2:
            fs = frozenset(support)
            prev = self.joint.get(fs)
            assert prev is None, f"joint teacher {sorted(fs)} already defined by '{prev['path']}'"
            self.joint[fs] = entry
        else:
            raise AssertionError(
                f"teacher '{path}' activates {len(support)} axes (>2); "
                f"the teacher target only supports up to 2 active axes"
            )
        self.entries.append(entry)
        self._by_path[key] = entry
        return entry["name"]

    def describe(self):
        lines = []
        for e in self.entries:
            n_active = sum(1 for x in e["vec"] if abs(x) > _DECOMP_EPS)
            kind = "solo" if n_active == 1 else f"joint({n_active} axes)"
            lines.append(f"  {e['name']} {kind}: vec={list(e['vec'])} <- {e['path']}")
        return "\n".join(lines)

    def decompose(self, omega):
        """Coordinate -> [(adapter_name | 'vanilla', weight), ...] (weights sum to 1)."""
        vals = [float(x) for x in omega]
        active = [(j, a) for j, a in enumerate(vals) if abs(a) > _DECOMP_EPS]
        if not active:
            return [("vanilla", 1.0)]

        if len(active) == 1:
            j, a = active[0]
            entry = self.solo.get(j)
            assert entry is not None, (
                f"coordinate {vals} uses axis {j}, but no solo teacher is registered for it"
            )
            out = []
            if a < 1.0 - _DECOMP_EPS:
                out.append(("vanilla", 1.0 - a))
            out.append((entry["name"], a))
            return out

        if len(active) == 2:
            support = frozenset(j for j, _ in active)
            entry_joint = self.joint.get(support)
            assert entry_joint is not None, (
                f"coordinate {vals} activates axes {sorted(support)}, but no joint "
                f"teacher for that pair is registered"
            )
            (hi_j, hi_a), (lo_j, lo_a) = sorted(active, key=lambda t: t[1], reverse=True)
            m = lo_a
            solo_hi = self.solo.get(hi_j)
            solo_lo = self.solo.get(lo_j)
            assert solo_hi is not None and solo_lo is not None, (
                f"joint teacher exists but member solo teachers are missing "
                f"(axis {hi_j}: {'ok' if solo_hi else 'missing'}, "
                f"axis {lo_j}: {'ok' if solo_lo else 'missing'})"
            )
            out = []
            w_base = 1.0 - hi_a - lo_a + m
            if w_base > _DECOMP_EPS:
                out.append(("vanilla", w_base))
            if hi_a - m > _DECOMP_EPS:
                out.append((solo_hi["name"], hi_a - m))
            if lo_a - m > _DECOMP_EPS:
                out.append((solo_lo["name"], lo_a - m))
            out.append((entry_joint["name"], m))
            return out

        raise AssertionError(
            f"coordinate {vals} activates {len(active)} axes; the teacher target "
            f"supports at most 2 active axes"
        )


def load_model_with_teachers(args, train_dtype, main_print):
    """Load FLUX with a trainable "student" LoRA plus frozen "teacher_N" adapters.

    "vanilla" is an optional zero LoRA (raw base model) used by coordinate mixtures.
    """
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=train_dtype,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    main_print(f"--> Creating student LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    transformer = get_peft_model(
        transformer,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            init_lora_weights="gaussian",
            lora_dropout=0.0,
            bias="none",
        ),
        adapter_name="student",
    )

    teacher_specs = getattr(args, "teacher_specs", None)
    assert teacher_specs, "args.teacher_specs is empty; build the TeacherRegistry first"
    for adapter_name, teacher_path in teacher_specs:
        if teacher_path is not None and str(teacher_path).strip().lower() == "none":
            # A "none" teacher is a zero-init LoRA whose forward equals the base model.
            main_print(f"--> Teacher {adapter_name}: 'none' -> zero LoRA (raw base model)")
            transformer.add_adapter(
                adapter_name,
                LoraConfig(
                    r=1,
                    lora_alpha=1,
                    target_modules=LORA_TARGET_MODULES,
                    init_lora_weights=True,
                    lora_dropout=0.0,
                    bias="none",
                ),
            )
        else:
            main_print(f"--> Loading teacher LoRA '{adapter_name}' from {teacher_path}")
            transformer.load_adapter(teacher_path, adapter_name=adapter_name, is_trainable=False)

    if getattr(args, "needs_vanilla", False):
        main_print("--> Adding frozen 'vanilla' adapter (zero LoRA = raw base model)")
        transformer.add_adapter(
            "vanilla",
            LoraConfig(
                r=1,
                lora_alpha=1,
                target_modules=LORA_TARGET_MODULES,
                init_lora_weights=True,
                lora_dropout=0.0,
                bias="none",
            ),
        )

    # Freeze everything except the student LoRA.
    frozen_names = {name for name, _ in teacher_specs} | {"vanilla"}
    for name, param in transformer.named_parameters():
        if any(f".{n}." in f".{name}." for n in frozen_names):
            param.requires_grad = False
        elif "lora" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    unwrap(transformer).set_adapter("student")
    main_print(f"--> Student LoRA ready with {len(teacher_specs)} teacher adapter(s)")
    return transformer


class CapFieldConditioner(nn.Module):
    """Two-layer MLP: coordinate [B, M] -> SiLU -> additive embedding [B, mod_dim].

    Bias-free with a zero-initialized output, so coordinate = 0 gives exactly zero.
    """

    def __init__(self, coord_dim, mod_dim, hidden_dim=256):
        super().__init__()
        self.in_proj = nn.Linear(coord_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, mod_dim, bias=False)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, omega):
        return self.out_proj(F.silu(self.in_proj(omega)))


class CapFieldHook:
    """Injects `CapFieldConditioner` into the FLUX global conditioning vector.

    The global vector is additive, so a forward hook on the module that builds it
    (`time_in` / `time_text_embed`) equals adding to the final vector.
    """

    def __init__(self, peft_transformer, cond_vectors, hidden_dim=256, main_print=print):
        flux, vec_module, vec_attr = self._find_flux_backbone(peft_transformer)
        cfg = flux.config
        mod_dim = getattr(cfg, "inner_dim", None)
        if mod_dim is None:
            mod_dim = cfg.num_attention_heads * cfg.attention_head_dim

        def _normalize_levels(entry):
            first = entry[0]
            if isinstance(first, (list, tuple)) or torch.is_tensor(first):
                return [[float(x) for x in lv] for lv in entry]
            return [[float(x) for x in entry]]

        self.cond_vectors = [_normalize_levels(v) for v in cond_vectors]
        self.cond_dim = len(self.cond_vectors[0][0])

        self.conditioner = CapFieldConditioner(self.cond_dim, mod_dim, hidden_dim)
        flux.add_module("task_conditioner", self.conditioner)

        self.enabled = False
        self._omega = None
        self.last_cond_matrix = None
        self.last_rollout_omega = None
        self.rollout_pools = None
        self.rollout_probs = None
        self.last_e_task_norm = 0.0
        self._handle = vec_module.register_forward_hook(self._hook)

        for si, levels in enumerate(self.cond_vectors):
            main_print(f"[CapField] source {si}: {len(levels)} level(s) {levels}")
        main_print(
            f"[CapField] coord_dim={self.cond_dim} mod_dim={mod_dim} hidden={hidden_dim} "
            f"(hook on flux.{vec_attr})"
        )

    @staticmethod
    def _find_flux_backbone(model):
        m = unwrap(model)
        for _ in range(8):
            for attr in ("time_in", "time_text_embed"):
                if hasattr(m, attr):
                    return m, getattr(m, attr), attr
            if hasattr(m, "model"):
                m = m.model
            elif hasattr(m, "base_model"):
                m = m.base_model
            else:
                break
        raise RuntimeError("cannot locate the FLUX backbone module (time_in / time_text_embed)")

    def _hook(self, module, inputs, output):
        if not self.enabled or self._omega is None:
            return output
        omega = self._omega.to(device=output.device, dtype=output.dtype)
        param = next(self.conditioner.parameters())
        if param.device != output.device:
            self.conditioner.to(device=output.device)
        e_task = self.conditioner(omega)
        self.last_e_task_norm = float(e_task.norm(dim=-1).mean())
        return output + e_task

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def set_coord(self, omega):
        """Inference API: set arbitrary coordinates, shape [M] or [B, M]."""
        t = torch.as_tensor(omega, dtype=torch.float32)
        if t.ndim == 1:
            t = t[None, :]
        self._omega = t

    def configure_rollout(self, rollout_pools, rollout_probs=None):
        """Set the per-axis rollout coordinate pools (None -> one-hot e_i).

        `rollout_probs[i]` are the matching sampling probabilities (None -> uniform).
        """
        self.rollout_pools = None
        self.rollout_probs = None
        if not rollout_pools:
            return

        assert len(rollout_pools) == self.cond_dim, (
            f"rollout pool count ({len(rollout_pools)}) != coordinate dim ({self.cond_dim})"
        )
        pools = []
        for pool in rollout_pools:
            if not pool:
                pools.append(None)
                continue
            levels = [[float(x) for x in v] for v in pool]
            for v in levels:
                assert len(v) == self.cond_dim, f"rollout coord {v} has wrong dim"
            pools.append(levels)
        if any(p is not None for p in pools):
            self.rollout_pools = pools

        if rollout_probs is not None:
            assert len(rollout_probs) == self.cond_dim, "rollout prob count mismatch"
            probs = []
            for si, (pool, pw) in enumerate(zip(self.rollout_pools, rollout_probs)):
                if not pool or not pw:
                    probs.append(None)
                    continue
                assert len(pw) == len(pool), f"source {si}: prob count != pool size"
                assert all(x > 0 for x in pw), f"source {si}: probabilities must be positive"
                total = sum(pw)
                probs.append([x / total for x in pw])
            if any(p is not None for p in probs):
                self.rollout_probs = probs

    def _draw_level_index(self, source_idx):
        pool = self.rollout_pools[source_idx]
        pw = self.rollout_probs[source_idx] if self.rollout_probs is not None else None
        if pw is not None:
            return int(torch.multinomial(torch.tensor(pw, dtype=torch.double), 1).item())
        return int(torch.randint(len(pool), (1,)).item())

    def set_rollout_batch(self, sources):
        """Sample one coordinate per prompt, reused across the whole rollout."""
        src = sources.tolist() if torch.is_tensor(sources) else [int(s) for s in sources]
        device = sources.device if torch.is_tensor(sources) else None
        cache = {}
        omegas = []
        for s in src:
            pool = self.rollout_pools[s] if self.rollout_pools is not None else None
            if not pool:
                e = [0.0] * self.cond_dim
                e[int(s)] = 1.0
                omegas.append(e)
            else:
                if s not in cache:
                    cache[s] = pool[self._draw_level_index(s)]
                omegas.append(list(cache[s]))
        self._omega = torch.tensor(omegas, dtype=torch.float32, device=device)
        self.last_cond_matrix = self._omega.detach().clone()
        self.last_rollout_omega = self._omega.detach().clone()

    def set_batch_from_sources(self, sources):
        """Fallback (no rollout linkage): draw the loss coordinate for every
        sample from the full per-source level pool, independently each timestep.

        Only used when `set_batch_from_rollout` has no rollout coordinate to
        link against (rollout_cond=None), matching the reference implementation.
        """
        src = sources.tolist() if torch.is_tensor(sources) else [int(s) for s in sources]
        omegas = []
        for s in src:
            levels = self.cond_vectors[int(s)]
            k = 0 if len(levels) == 1 else int(torch.randint(len(levels), (1,)).item())
            omegas.append(list(levels[k]))
        self._omega = torch.tensor(omegas, dtype=torch.float32, device=sources.device)
        self.last_cond_matrix = self._omega.detach().clone()

    def set_batch_from_rollout(self, sources, rollout_cond):
        """Draw the per-timestep coordinate from the source pool.

        Only levels whose active-axis support matches the rollout coordinate are
        kept (the all-zero level is a wildcard); falls back to the rollout
        coordinate itself if the sub-pool is empty.
        """
        src = sources.tolist() if torch.is_tensor(sources) else [int(s) for s in sources]
        if torch.is_tensor(rollout_cond):
            conds = rollout_cond.detach().float().cpu().tolist()
        else:
            conds = [list(c) for c in rollout_cond]

        omegas = []
        for s, c in zip(src, conds):
            support = frozenset(j for j, v in enumerate(c) if abs(v) > _DECOMP_EPS)
            sub = []
            for lv in self.cond_vectors[int(s)]:
                lv_support = frozenset(j for j, v in enumerate(lv) if abs(v) > _DECOMP_EPS)
                if lv_support == support or not lv_support:
                    sub.append(lv)
            if sub:
                omegas.append(list(sub[int(torch.randint(len(sub), (1,)).item())]))
            else:
                omegas.append(list(c))
        self._omega = torch.tensor(omegas, dtype=torch.float32, device=sources.device)
        self.last_cond_matrix = self._omega.detach().clone()

    def save(self, save_dir):
        flat, seen = [], set()
        for levels in self.cond_vectors:
            for lv in levels:
                key = tuple(lv)
                if key not in seen:
                    seen.add(key)
                    flat.append(lv)
        meta = {
            "cond_vectors": flat,
            "cond_levels": self.cond_vectors,
            "cond_dim": self.cond_dim,
            "hidden_dim": self.conditioner.in_proj.out_features,
        }
        torch.save(
            {"state_dict": self.conditioner.state_dict(), "meta": meta},
            os.path.join(save_dir, CONDITIONER_FILENAMES[0]),
        )

    def load(self, load_dir):
        ckpt_path = find_conditioner_file(load_dir)
        if ckpt_path is None:
            return False
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.conditioner.load_state_dict(ckpt["state_dict"])
        return True
