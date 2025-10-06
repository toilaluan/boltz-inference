from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Union

import os
import inspect
import dataclasses

import torch
from torch import Tensor, nn
from torch.profiler import ProfilerActivity, profile

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.encodersv2 import RelativePositionEncoder
from boltz.model.modules.trunkv2 import (
    ContactConditioning,
    DistogramModule,
    InputEmbedder,
    MSAModule,
    TemplateModule,
    TemplateV2Module,
)


class Boltz2TrunkInfer(nn.Module):

    def __init__(
        self,
        *,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        embedder_args: Dict[str, Any],
        msa_args: Dict[str, Any],
        pairformer_args: Dict[str, Any],
        # Trunk options kept explicit to avoid implicit defaults
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        bond_type_feature: bool = False,
        use_templates: bool = False,
        compile_pairformer: bool = False,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
    ) -> None:
        super().__init__()

        # Make compile caches explicit for big graphs
        torch._dynamo.config.cache_size_limit = 512  # type: ignore[attr-defined]
        torch._dynamo.config.accumulated_cache_size_limit = 512  # type: ignore[attr-defined]

        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.use_templates = use_templates
        self.bond_type_feature = bond_type_feature
        self.compile_pairformer = compile_pairformer

        # ----- Embeddings -----
        self.input_embedder = InputEmbedder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
            **embedder_args,
        )

        self.s_init = nn.Linear(token_s, token_s, bias=False)
        self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.z_init_2 = nn.Linear(token_s, token_z, bias=False)

        self.rel_pos = RelativePositionEncoder(
            token_z, fix_sym_check=fix_sym_check, cyclic_pos_enc=cyclic_pos_enc
        )

        self.token_bonds = nn.Linear(1, token_z, bias=False)
        if bond_type_feature:
            self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

        self.contact_conditioning = ContactConditioning(
            token_z=token_z,
            cutoff_min=conditioning_cutoff_min,
            cutoff_max=conditioning_cutoff_max,
        )

        # Recycle + norm
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # Trunk modules
        self.msa_module = MSAModule(token_z=token_z, token_s=token_s, **msa_args)
        self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)

        # Heads
        self.distogram_module = DistogramModule(token_z, num_bins)

        # Optional per-layer compile to avoid graph blow-ups
        if self.compile_pairformer:
            for i in range(len(self.pairformer_module.layers)):
                self.pairformer_module.layers[i] = torch.compile(  # type: ignore[assignment]
                    self.pairformer_module.layers[i],
                    dynamic=False,
                    fullgraph=False,
                )

    # --- Core: chunk inference only ---
    def forward_trunk(self, feats: Dict[str, Tensor], *, recycling_steps: int = 0) -> Dict[str, Tensor]:
        s_inputs = self.input_embedder(feats)

        s_init = self.s_init(s_inputs)
        z_init = self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]

        rpe = self.rel_pos(feats)
        z = z_init + rpe
        z = z + self.token_bonds(feats["token_bonds"].float())
        if self.bond_type_feature:
            z = z + self.token_bonds_type(feats["type_bonds"].long())
        z = z + self.contact_conditioning(feats)

        s = torch.zeros_like(s_init)

        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        for _ in range(recycling_steps + 1):
            s = s_init + self.s_recycle(self.s_norm(s))
            z = z + self.z_recycle(self.z_norm(z))  # explicit add to keep flow obvious

            z = z + self.msa_module(z, s_inputs, feats)

            s, z = self.pairformer_module(
                s, z, mask=mask, pair_mask=pair_mask
            )

        pdistogram = self.distogram_module(z)

        return {
            "s": s,
            "z": z,
            "pdistogram": pdistogram,
        }

    # Alias to keep callers simple
    def forward(self, feats: Dict[str, Tensor], *, recycling_steps: int = 0) -> Dict[str, Tensor]:
        return self.forward_trunk(feats, recycling_steps=recycling_steps)

    # ---- Light-weight checkpoint loader (no Lightning required) ----
    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_path: str,
        strict: bool = True,
        map_location: Optional[Union[str, torch.device]] = None,
        **overrides: Any,
    ) -> "Boltz2TrunkInfer":
        path = os.path.expanduser(ckpt_path)
        ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            hparams: Dict[str, Any] = ckpt.get("hyper_parameters", {}) or ckpt.get("hparams", {}) or {}
        else:
            state_dict = ckpt
            hparams = {}

        def to_plain(x: Any) -> Any:
            if dataclasses.is_dataclass(x):
                return dataclasses.asdict(x)
            if hasattr(x, "to_container"):
                try:
                    return x.to_container(resolve=True)  # OmegaConf
                except Exception:
                    return x
            return x

        hparams = {k: to_plain(v) for k, v in hparams.items()}

        sig = inspect.signature(cls.__init__)
        allowed = {
            p.name
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name != "self"
        }

        # Only pass through args this slim class actually uses
        init_kwargs: Dict[str, Any] = {k: v for k, v in hparams.items() if k in allowed}
        for k, v in overrides.items():
            if k in allowed:
                init_kwargs[k] = v

        required = [
            "atom_s",
            "atom_z",
            "token_s",
            "token_z",
            "num_bins",
            "embedder_args",
            "msa_args",
            "pairformer_args",
            "atoms_per_window_queries",
            "atoms_per_window_keys",
            "atom_feature_dim",
        ]
        missing = [k for k in required if k not in init_kwargs]
        if missing:
            raise KeyError(f"Missing required init args in checkpoint/overrides: {missing}")

        model = cls(**init_kwargs)

        def normalize_key(k: str) -> str:
            if k.startswith("model."):
                k = k[6:]
            return k.replace("._orig_mod", "")

        model_state = model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
        model.load_state_dict(filtered_state, strict=strict)

        if map_location is not None:
            device = torch.device(map_location) if isinstance(map_location, str) else map_location
            model.to(device)

        # Persist non-ctor user data (ignored by this slim class anyway, but harmless)
        for k, v in overrides.items():
            if k not in allowed:
                setattr(model, k, v)

        model.eval()
        return model


# --- Optional: quick profiler for trunk only ---
def profile_chunk_inference(
    model: Boltz2TrunkInfer,
    feats: Dict[str, Tensor],
    *,
    compile_pairformer_layers: bool = False,
    trace_path: str = "trace.json",
    recycling_steps: int = 0,
    profiler_kwargs: Optional[Dict[str, Any]] = None,
    row_limit: int = 25,
) -> Dict[str, Tensor]:
    """
    Profile chunk inference with torch.profiler and emit a Chrome trace.
    """
    if compile_pairformer_layers:
        for i in range(len(model.pairformer_module.layers)):
            model.pairformer_module.layers[i] = torch.compile(  # type: ignore[assignment]
                model.pairformer_module.layers[i],
                dynamic=False,
                fullgraph=False,
            )

    kwargs = dict(profiler_kwargs or {})
    activities: Optional[Sequence[ProfilerActivity]] = kwargs.pop("activities", None)
    if activities is None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

    with torch.inference_mode():
        _ = model.forward_trunk(feats, recycling_steps=recycling_steps)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    defaults = {
        "activities": activities,
        "record_shapes": True,
        "with_stack": True,
        "profile_memory": True,
    }
    defaults.update(kwargs)

    with profile(**defaults) as prof:
        with torch.inference_mode():
            out = model.forward_trunk(feats, recycling_steps=recycling_steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prof.step()

    averages = prof.key_averages()
    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    print(averages.table(sort_by=sort_key, row_limit=row_limit))
    if torch.cuda.is_available():
        print(averages.table(sort_by="self_cpu_time_total", row_limit=row_limit))

    if trace_path:
        d = os.path.dirname(os.fspath(trace_path))
        if d:
            os.makedirs(d, exist_ok=True)
        prof.export_chrome_trace(os.fspath(trace_path))

    return out
