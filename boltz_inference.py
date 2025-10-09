from __future__ import annotations

from typing import Any, Optional, Dict, Union
import dataclasses
import inspect
import os
import json


import torch
from torch import nn, Tensor

# --- Boltz2 building blocks (kept minimal and inference-only) ---
import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.modules.encodersv2 import RelativePositionEncoder
from boltz.model.modules.trunkv2 import (
    InputEmbedder,
    MSAModule,
    DistogramModule,
    ContactConditioning,
    TemplateModule,
    TemplateV2Module,
)
from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
from boltz.model.modules.diffusionv2 import AtomDiffusion
from boltz.model.modules.confidencev2 import ConfidenceModule
from boltz.model.modules.affinity import AffinityModule


class Boltz2AffinityInference(nn.Module):
    """
    Minimal, inference-only module that computes `affinity_probability_binary`.

    Kept components (required for correctness):
      - Input embedding + relative/conditioning features
      - MSA + Pairformer trunk to produce (s, z)
      - Distogram head (logits are consumed by ConfidenceModule)
      - DiffusionConditioning + AtomDiffusion to get predicted coordinates
      - ConfidenceModule to compute iPTM and pick the best sample
      - AffinityModule to produce binary affinity logits -> probability

    Removed: training, schedulers, logging, validation, EMA, losses, etc.
    """

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
        score_model_args: Dict[str, Any],
        diffusion_process_args: Dict[str, Any],
        confidence_model_args: Dict[str, Any],
        affinity_model_args1: Dict[str, Any],
        affinity_model_args2: Dict[str, Any],
        template_args: Dict[str, Any],
        steering_args: Dict[str, Any],
        # Flags / options
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_feature_dim: int = 128,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        use_kernels: bool = False,
    ) -> None:
        super().__init__()
        self.steering_args = steering_args
        # --- Embedders & small feature heads ---
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            "use_no_atom_char": use_no_atom_char,
            "use_atom_backbone_feat": use_atom_backbone_feat,
            "use_residue_feats_atoms": use_residue_feats_atoms,
            **embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)

        self.s_init = nn.Linear(token_s, token_s, bias=False)
        self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.z_init_2 = nn.Linear(token_s, token_z, bias=False)

        self.rel_pos = RelativePositionEncoder(
            token_z, fix_sym_check=fix_sym_check, cyclic_pos_enc=cyclic_pos_enc
        )

        self.token_bonds = nn.Linear(1, token_z, bias=False)
        self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

        self.contact_conditioning = ContactConditioning(
            token_z=token_z,
            cutoff_min=conditioning_cutoff_min,
            cutoff_max=conditioning_cutoff_max,
        )

        # --- Simple recycling projections (single-pass) ---
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # --- Trunk ---
        self.msa_module = MSAModule(token_z=token_z, token_s=token_s, **msa_args)
        self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)

        # --- Heads / downstream modules ---
        self.distogram_module = DistogramModule(token_z, num_bins)

        self.diffusion_conditioning = DiffusionConditioning(
            token_s=token_s,
            token_z=token_z,
            atom_s=atom_s,
            atom_z=atom_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=score_model_args["atom_encoder_depth"],
            atom_encoder_heads=score_model_args["atom_encoder_heads"],
            token_transformer_depth=score_model_args["token_transformer_depth"],
            token_transformer_heads=score_model_args["token_transformer_heads"],
            atom_decoder_depth=score_model_args["atom_decoder_depth"],
            atom_decoder_heads=score_model_args["atom_decoder_heads"],
            atom_feature_dim=atom_feature_dim,
            conditioning_transition_layers=score_model_args["conditioning_transition_layers"],
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )

        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_s": token_s,
                "atom_s": atom_s,
                "atoms_per_window_queries": atoms_per_window_queries,
                "atoms_per_window_keys": atoms_per_window_keys,
                **score_model_args,
            },
            **diffusion_process_args,
        )

        self.confidence_module = ConfidenceModule(
            token_s,
            token_z,
            token_level_confidence=True,
            bond_type_feature=True,
            fix_sym_check=fix_sym_check,
            cyclic_pos_enc=cyclic_pos_enc,
            conditioning_cutoff_min=conditioning_cutoff_min,
            conditioning_cutoff_max=conditioning_cutoff_max,
            **confidence_model_args,
        )

        self.affinity_module1 = AffinityModule(
            token_s,
            token_z,
            **affinity_model_args1,
        )
        self.affinity_module2 = AffinityModule(
            token_s,
            token_z,
            **affinity_model_args2,
        )

        # --- misc flags ---
        self.use_kernels = use_kernels

    @torch.no_grad()
    def forward(
        self,
        feats: Dict[str, Tensor],
        *,
        recycling_steps: int = 0,  # set 0 for leanest pass
        diffusion_samples: int = 1,
        num_sampling_steps: Optional[int] = None,
        run_confidence_sequentially: bool = True,
        use_iptm_selection: bool = True,
        max_parallel_samples: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Compute affinity probability from a single forward call.

        Expected keys in `feats` (besides whatever your embedder needs):
          - token_bonds, token_pad_mask, atom_pad_mask
          - mol_type, affinity_token_mask
          - (optional) type_bonds if bond_type_feature=True
        """
        # --- Input embeddings ---
        s_inputs = self.input_embedder(feats)  # (B, L, token_s)
        torch.save(s_inputs, "s_inputs1.pt")

        # Initialize sequence & pair embeddings
        s_init = self.s_init(s_inputs)
        z_init = self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        z_init = z_init + self.rel_pos(feats)
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(feats)

        # Compute masks
        mask = feats["token_pad_mask"].float()                  # (B, L)
        pair_mask = mask[:, :, None] * mask[:, None, :]         # (B, L, L)

        # Single pass (no multi-recycling): start from zeros like the original and apply a single recycle step
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        for i in range(recycling_steps+1):
            print(f"Recycling step {i}/{recycling_steps}")
            s = s_init + self.s_recycle(self.s_norm(s))
            z = z_init + self.z_recycle(self.z_norm(z))

            # Trunk
            z = z + self.msa_module(z, s_inputs, feats)
            s, z = self.pairformer_module(
                s, z, mask=mask, pair_mask=pair_mask
            )

        # Distogram logits (Confidence module consumes one distogram head)
        pdistogram = self.distogram_module(z)

        # Diffusion conditioning + structure sampling (coordinates)
        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = self.diffusion_conditioning(
            s_trunk=s,
            z_trunk=z,
            relative_position_encoding=self.rel_pos(feats),
            feats=feats,
        )
        diffusion_conditioning = {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }

        with torch.autocast("cuda", enabled=False):
            struct_out = self.structure_module.sample(
                s_trunk=s.float(),
                s_inputs=s_inputs.float(),
                feats=feats,
                num_sampling_steps=num_sampling_steps,
                atom_mask=feats["atom_pad_mask"].float(),
                multiplicity=diffusion_samples,
                max_parallel_samples=max_parallel_samples,
                steering_args=self.steering_args,
                diffusion_conditioning=diffusion_conditioning,
            )
        sample_atom_coords = struct_out["sample_atom_coords"]  # (S, K=1, L, 3) typically

        # Confidence to get iPTM and choose best sample
        conf_out = self.confidence_module(
            s_inputs=s_inputs.detach(),
            s=s.detach(),
            z=z.detach(),
            x_pred=sample_atom_coords.detach(),
            feats=feats,
            pred_distogram_logits=pdistogram[:, :, :, 0].detach(),  # consume one head
            multiplicity=diffusion_samples,
            run_sequentially=True,
            use_kernels=True,
        )

        best_idx = torch.argsort(conf_out["iptm"], descending=True)[0].item()

        coords_affinity = sample_atom_coords.detach()[best_idx][None, None]  # (1,1,L,3)

        # Affinity-specific masking on pair reps
        pad_token_mask = feats["token_pad_mask"][0]
        rec_mask = (feats["mol_type"][0] == 0) * pad_token_mask
        lig_mask = feats["affinity_token_mask"][0].to(torch.bool) * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, None] * rec_mask[None, :] +
            rec_mask[:, None] * lig_mask[None, :] +
            lig_mask[:, None] * lig_mask[None, :]
        )  # (L, L)
        z_affinity = z * cross_pair_mask[None, :, :, None]

        # Affinity-specific input embedding (if your embedder uses a special path)
        s_inputs_aff = self.input_embedder(feats)

        with torch.autocast("cuda", enabled=False):
            aff_out1 = self.affinity_module1(
                s_inputs=s_inputs_aff.detach(),
                z=z_affinity.detach(),
                x_pred=coords_affinity,
                feats=feats,
                multiplicity=1,
            )
            aff_out2 = self.affinity_module2(
                s_inputs=s_inputs_aff.detach(),
                z=z_affinity.detach(),
                x_pred=coords_affinity,
                feats=feats,
                multiplicity=1,
            )
            prob = torch.nn.functional.sigmoid(aff_out1["affinity_logits_binary"]) + torch.nn.functional.sigmoid(aff_out2["affinity_logits_binary"])
            prob = prob / 2.0

        return prob
    # ------------------------------------------------------------------
    # Convenience: load a pretrained checkpoint (strict=False)
    # ------------------------------------------------------------------
    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_path: str,
        strict: bool = True,
        map_location: Optional[Union[str, torch.device]] = None,
        **overrides: Any,
    ) -> "Boltz2AffinityInference":
        path = os.path.expanduser(ckpt_path)
        ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        with open("hparams.json", "r") as f:
            hparams = json.load(f)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()
        model.to("cuda")
        return model



# Optional: a simple wrapper function
@torch.no_grad()
def predict_affinity_probability(
    model: Boltz2AffinityInference,
    feats: Dict[str, Tensor],
    *,
    recycling_steps: int = 0,
    diffusion_samples: int = 1,
    num_sampling_steps: Optional[int] = None,
) -> Tensor:
    out = model(
        feats,
        recycling_steps=recycling_steps,
        diffusion_samples=diffusion_samples,
        num_sampling_steps=num_sampling_steps,
        run_confidence_sequentially=True,
        use_iptm_selection=True,
        max_parallel_samples=None,
    )
    return out


