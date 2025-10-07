#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import asdict, dataclass
import numpy as np
import torch

from boltz.data import const
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.yaml import parse_yaml
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import (
    MSA,
    Input,
    Record,
    ResidueConstraints,
    StructureV2,
)
from boltz.data.pad import pad_to_max

# -----------------------------
# Public API
# -----------------------------

def preprocess(
    item_yaml_file: str | Path,
    *,
    cache_dir: str | Path = "~/.boltz",
    affinity: bool = False,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Preprocess a single YAML item into model-ready features, matching PredictionDataset.__getitem__.

    Parameters
    ----------
    item_yaml_file
        Path to a single input YAML file.
    cache_dir
        Directory containing Boltz canonical molecules (expected at <cache_dir>/mols).
    use_msa_server
        If True, generate MSAs for protein chains with missing MSAs using an MMseqs2 server.
    msa_server_url, msa_pairing_strategy, msa_server_username, msa_server_password, api_key_header, api_key_value
        MMseqs2 server options (mirrors your pipeline).
    override_method
        Optional featurizer override_method (same as dataset).
    affinity
        If True, apply AffinityCropper prior to featurization and enable affinity features.
    seed
        RNG seed for featurization randomness.
    max_msa_seqs
        Maximum MSA sequences to keep per chain (parsing limit).
    crop_max_tokens, crop_max_atoms
        Affinity cropping bounds.

    Returns
    -------
    Dict[str, Tensor]
        Features dict identical to what your dataloader returns for one item (plus "record").
    """
    torch.set_grad_enabled(False)

    cache_dir = Path(cache_dir).expanduser()
    mol_dir = cache_dir / "mols"

    # 1) Parse YAML into a Target-like object (structure, record, constraints, templates, extra_mols, sequences)
    #    For boltz2 path, pass canonical mols as "ccd" to yaml parser (this is how your pipeline does it).
    canonicals = load_canonicals(mol_dir)
    target = parse_yaml(Path(item_yaml_file), canonicals, mol_dir, boltz2=True)
    record: Record = target.record
    # structure: StructureV2 = target.structure
    structure = StructureV2.load("/workspace/boltz-inference/outputs/boltz_results_test_samples/predictions/P23975_ligand_0029/pre_affinity_P23975_ligand_0029.npz")
    # print(structure2)
    # print(structure)
    residue_constraints: Optional[ResidueConstraints] = target.residue_constraints
    templates: Dict[str, StructureV2] = target.templates or {}
    extra_mols: Dict = target.extra_mols or {}
    sequences: Dict[int, str] = target.sequences or {}


    # 3) Construct Input object directly (no disk round-trips)
    input_data = Input(
        structure,
        {},
        record=record,
        residue_constraints=residue_constraints,
        templates=templates if templates else None,
        extra_mols=extra_mols,
    )

    # 4) Tokenize (optionally crop for affinity)
    tokenizer = Boltz2Tokenizer()
    tokenized = tokenizer.tokenize(input_data)

    if affinity:
        cropper = AffinityCropper()
        tokenized = cropper.crop(
            tokenized,
            max_tokens=256,
            max_atoms=2048,
        )

    # 5) Load molecules: canonicals + extra + any missing ones referenced by tokens
    molecules = dict(canonicals)
    molecules.update(extra_mols)

    res_names = set(tokenized.tokens["res_name"].tolist())
    missing = res_names - set(molecules.keys())
    if missing:
        molecules.update(load_molecules(mol_dir, missing))

    # 6) Featurize with the same settings as your dataset
    rng = np.random.default_rng(seed)
    options = record.inference_options
    if options is None:
        pocket_constraints = None
        contact_constraints = None
    else:
        pocket_constraints = options.pocket_constraints
        contact_constraints = options.contact_constraints
    seed = 42
    random = np.random.default_rng(seed)
    featurizer = Boltz2Featurizer()
    features = featurizer.process(
        tokenized,
        molecules=molecules,
        random=random,
        training=False,
        max_atoms=None,
        max_tokens=None,
        max_seqs=const.max_msa_seqs,
        pad_to_max_seqs=False,
        single_sequence_prop=0.0,
        compute_frames=True,
        inference_pocket_constraints=pocket_constraints,
        inference_contact_constraints=contact_constraints,
        compute_constraint_features=True,
        override_method="other",
        compute_affinity=affinity,
    )

    features["record"] = record
    return features



def _is_all_gaps(s: str) -> bool:
    return s and set(s) == {"-"}



def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "affinity_mw",
        ]:
            # Check if all have the same shape
            shape_report = []
            for v in values:
                shape_report.append(v.shape)
            print(key, shape_report)
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated

def transfer_batch_to_device(
    batch: dict,
    device: torch.device,
) -> dict:
    """Transfer a batch to the given device.

    Parameters
    ----------
    batch : Dict
        The batch to transfer.
    device : torch.device
        The device to transfer to.
    dataloader_idx : int
        The dataloader index.

    Returns
    -------
    np.Any
        The transferred batch.

    """
    for key in batch:
        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "affinity_mw",
        ]:
            batch[key] = batch[key].to(device)
    return batch

@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None


@dataclass
class PairformerArgs:
    """Pairformer arguments."""

    num_blocks: int = 48
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = False


@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = False
    alignment_reverse_diff: bool = False
    synchronize_sigmas: bool = True


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    contact_guidance_update: bool = True
    num_gd_steps: int = 20

if __name__ == "__main__":
    import time
    from dataclasses import asdict, dataclass

    from torch.profiler import ProfilerActivity, profile

    parser = argparse.ArgumentParser(
        description="Profile Boltz inference with torch.compile."
    )
    parser.add_argument(
        "--input-yaml",
        default="samples/P23975_ligand_0029.yaml",
        help="Path to the input YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="~/.boltz/boltz2_aff.ckpt",
        help="Checkpoint path for Boltz2ChunkInfer.",
    )
    parser.add_argument(
        "--compile-modes",
        nargs="+",
        default=["default"],
        help="Torch compile modes to evaluate. Use 'eager' to skip compilation.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Number of warmup iterations before profiling.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=1,
        help="Number of iterations to capture in the profiler.",
    )
    parser.add_argument(
        "--recycling-steps",
        type=int,
        default=0,
        help="Recycling steps to use during inference.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=25,
        help="Row limit when printing profiler statistics.",
    )
    args = parser.parse_args()


    start = time.time()
    features = preprocess(args.input_yaml, affinity=True)
    features = collate([features])
    preprocess_elapsed = time.time() - start
    print(f"Preprocessing finished in {preprocess_elapsed:.2f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = transfer_batch_to_device(features, device)

    # from trunk_inference_model import Boltz2TrunkInfer
    from boltz_inference import Boltz2AffinityInference

    subsample_msa = True
    num_subsampled_msa = 1024
    sampling_steps = 100
    diffusion_samples = 1
    max_parallel_samples = 1
    write_full_pae = False
    write_full_pde = False
    use_potentials = False

    def build_model() -> Boltz2AffinityInference:
        diffusion_params = Boltz2DiffusionParams()
        diffusion_params.step_scale = 1.5
        pairformer_args = PairformerArgsV2()

        msa_args = MSAModuleArgs(
            subsample_msa=subsample_msa,
            num_subsampled_msa=num_subsampled_msa,
            use_paired_feature=True,
        )
        predict_args = {
            "recycling_steps": args.recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": write_full_pae,
            "write_full_pde": write_full_pde,
        }

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.guidance_update = False
        steering_args.physical_guidance_update = False
        steering_args.contact_guidance_update = False

        model = Boltz2AffinityInference.load_from_checkpoint(
            args.checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location=device,
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        return model

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    compile_modes = args.compile_modes or ["default"]
    should_print_model = False

    def run_single_step(active_model: Boltz2AffinityInference) -> None:
        with torch.inference_mode():
            out = active_model(batch, recycling_steps=5, num_sampling_steps=100, diffusion_samples=3)
            print(out)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return out

    for mode in compile_modes:
        model = build_model()
        if should_print_model:
            print(model)
            should_print_model = False

        model = model
        compile_elapsed = 0.0

        compile_start = time.time()
        for layer in model.pairformer_module.layers:
            layer = torch.compile(
                layer,
                dynamic=False,
            )
        print(f"[compile:{mode}] torch.compile finished for pairformer in {compile_elapsed:.2f}s")
        for layer in model.msa_module.layers:
            layer = torch.compile(
                layer,
                dynamic=False,
            )
        print(f"[compile:{mode}] torch.compile finished for msa in {compile_elapsed:.2f}s")
        compile_elapsed = time.time() - compile_start
        print(f"[compile:{mode}] torch.compile finished in {compile_elapsed:.2f}s")

        warmup_elapsed = 0.0
        model.eval()
        if args.warmup_iters > 0:
            warmup_start = time.time()
            for _ in range(args.warmup_iters):
                _ = run_single_step(model)
            warmup_elapsed = time.time() - warmup_start
        print(f"[warmup:{mode}] {args.warmup_iters} iteration(s) in {warmup_elapsed:.2f}s")

        if args.profile_iters <= 0:
            raise ValueError("--profile-iters must be greater than 0 to collect profiling data.")

        profile_start = time.time()
        with profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:
            for _ in range(args.profile_iters):
                out = run_single_step(model)
                prof.step()
        profile_elapsed = time.time() - profile_start
        print(
            f"[profile:{mode}] collected {args.profile_iters} iteration(s) in {profile_elapsed:.2f}s"
        )

        averages = prof.key_averages()
        sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        print(f"[profile:{mode}] key averages sorted by {sort_key}")
        print(averages.table(sort_by=sort_key, row_limit=args.row_limit))
        if device.type == "cuda":
            print(averages.table(sort_by="self_cpu_time_total", row_limit=args.row_limit))

        trace_path = Path(f"trace-json-{mode}.json")
        prof.export_chrome_trace(trace_path.as_posix())
        print(f"[profile:{mode}] Chrome trace exported to {trace_path.resolve()}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(out)