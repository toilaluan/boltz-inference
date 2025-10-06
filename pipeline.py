#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    override_method: Optional[str] = None,
    affinity: bool = False,
    seed: int = 42,
    max_msa_seqs: int = 8192,
    crop_max_tokens: int = 256,
    crop_max_atoms: int = 2048,
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
    structure: StructureV2 = target.structure
    residue_constraints: Optional[ResidueConstraints] = target.residue_constraints
    templates: Dict[str, StructureV2] = target.templates or {}
    extra_mols: Dict = target.extra_mols or {}
    sequences: Dict[int, str] = target.sequences or {}

    # 2) Build MSAs in-memory for each protein chain as needed
    msas: Dict[int, MSA] = _resolve_msas_for_record(
        record=record,
        sequences=sequences,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        max_msa_seqs=max_msa_seqs,
    )

    # 3) Construct Input object directly (no disk round-trips)
    input_data = Input(
        structure,
        msas,
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
            max_tokens=crop_max_tokens,
            max_atoms=crop_max_atoms,
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

    featurizer = Boltz2Featurizer()
    features = featurizer.process(
        tokenized,
        molecules=molecules,
        random=rng,
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
        override_method=override_method,
        compute_affinity=affinity,
    )

    features["record"] = record
    return features


# -----------------------------
# Internals
# -----------------------------

def _resolve_msas_for_record(
    *,
    record: Record,
    sequences: Dict[int, str],
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str],
    msa_server_password: Optional[str],
    api_key_header: Optional[str],
    api_key_value: Optional[str],
    max_msa_seqs: int,
) -> Dict[int, MSA]:
    """
    Return a dict chain_id -> MSA, computing any missing MSAs via server if requested.
    Supports three cases:
      - chain.msa_id == -1: no MSA for this chain (skip)
      - chain.msa_id == 0 and chain is protein: generate via MMseqs2 server
      - chain.msa_id is a path to .a3m or .csv: load and parse
    """
    msas: Dict[int, MSA] = {}
    prot_id = const.chain_type_ids["PROTEIN"]

    # Pre-build auth headers (optional)
    auth_headers = None
    if api_key_value:
        key = api_key_header or "X-API-Key"
        auth_headers = {"Content-Type": "application/json", key: api_key_value}

    for chain in record.chains:
        if chain.msa_id == -1:
            continue

        if isinstance(chain.msa_id, (str, Path)):
            # Load an existing MSA file provided by YAML
            msa_path = Path(chain.msa_id)
            if not msa_path.exists():
                raise FileNotFoundError(f"MSA file {msa_path} not found for chain {chain.chain_id}")
            if msa_path.suffix == ".a3m":
                msa_obj = parse_a3m(msa_path, taxonomy=None, max_seqs=max_msa_seqs)
            elif msa_path.suffix == ".csv":
                msa_obj = parse_csv(msa_path, max_seqs=max_msa_seqs)
            else:
                raise RuntimeError(f"MSA file {msa_path} not supported (only .a3m or .csv).")
            msas[chain.chain_id] = msa_obj
            continue

        if chain.msa_id == 0 and chain.mol_type == prot_id:
            if not use_msa_server:
                raise RuntimeError(
                    f"Missing MSA for protein chain {chain.chain_id} "
                    f"and use_msa_server=False."
                )

            seq = sequences.get(chain.entity_id)
            if not seq or not isinstance(seq, str) or not len(seq):
                raise RuntimeError(
                    f"No protein sequence found for entity {chain.entity_id} "
                    f"(needed to compute MSA)."
                )

            # Compute paired & unpaired MSAs via server
            paired_list, unpaired_list = _mmseqs2_fetch(
                [seq],
                msa_server_url=msa_server_url,
                pairing_strategy=msa_pairing_strategy,
                msa_server_username=msa_server_username,
                msa_server_password=msa_server_password,
                auth_headers=auth_headers,
            )

            # Build a single CSV in-memory and parse it to MSA
            csv_text = _compose_csv_from_paired_unpaired(
                paired_list[0], unpaired_list[0], max_total=const.max_msa_seqs
            )
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                tmp.write(csv_text)
                tmp_path = Path(tmp.name)

            try:
                msa_obj = parse_csv(tmp_path, max_seqs=max_msa_seqs)
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

            msas[chain.chain_id] = msa_obj

    return msas


def _mmseqs2_fetch(
    seqs: list[str],
    *,
    msa_server_url: str,
    pairing_strategy: str,
    msa_server_username: Optional[str],
    msa_server_password: Optional[str],
    auth_headers: Optional[Dict[str, str]],
) -> Tuple[list[str], list[str]]:
    """
    Call MMseqs2 server similarly to your pipeline:
      - paired_msas: run with use_pairing=True
      - unpaired_msas: run with use_pairing=False
    Returns the raw FASTA/A3M-ish texts as lists of strings (aligned with input seqs).
    """
    # Note: run_mmseqs2 returns strings with alternating header/sequence lines when use_env=True
    if len(seqs) > 1:
        paired_msas = run_mmseqs2(
            seqs,
            out_dir=Path(tempfile.mkdtemp(prefix="boltz_msa_paired_")),
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            auth_headers=auth_headers,
        )
    else:
        paired_msas = [""] * len(seqs)

    unpaired_msas = run_mmseqs2(
        seqs,
        out_dir=Path(tempfile.mkdtemp(prefix="boltz_msa_unpaired_")),
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        auth_headers=auth_headers,
    )
    return paired_msas, unpaired_msas


def _compose_csv_from_paired_unpaired(
    paired_raw: str,
    unpaired_raw: str,
    *,
    max_total: int,
) -> str:
    """
    Build the same 'key,sequence' CSV used downstream, limiting to max_total sequences.
    Paired data: take every 2nd line (sequence only), drop all-gap rows, and cap.
    Unpaired data: take every 2nd line, cap to remaining budget (minus 1 if paired had query).
    """
    # Paired block
    paired_lines = [ln for ln in paired_raw.strip().splitlines() if ln]
    paired = paired_lines[1::2] if paired_lines else []
    # Remove all-gap sequences
    paired = [s for s in paired if s and not _is_all_gaps(s)]
    paired = paired[: const.max_paired_seqs]
    paired_keys = list(range(len(paired)))

    # Unpaired block
    unpaired_lines = [ln for ln in unpaired_raw.strip().splitlines() if ln]
    unpaired = unpaired_lines[1::2] if unpaired_lines else []

    # If paired present, drop the query from unpaired to avoid duplication
    if paired:
        unpaired = unpaired[1:] if len(unpaired) > 0 else unpaired

    budget = max_total - len(paired)
    if budget < 0:
        budget = 0
    unpaired = unpaired[:budget]
    unpaired_keys = [-1] * len(unpaired)

    seqs = paired + unpaired
    keys = paired_keys + unpaired_keys

    rows = ["key,sequence"]
    rows += [f"{k},{s}" for k, s in zip(keys, seqs)]
    return "\n".join(rows)


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
        default="~/.boltz/boltz2_conf.ckpt",
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

    start = time.time()
    features = preprocess(args.input_yaml)
    features = collate([features])
    preprocess_elapsed = time.time() - start
    print(f"Preprocessing finished in {preprocess_elapsed:.2f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = transfer_batch_to_device(features, device)

    from trunk_inference_model import Boltz2TrunkInfer

    subsample_msa = True
    num_subsampled_msa = 1024
    sampling_steps = 100
    diffusion_samples = 1
    max_parallel_samples = 1
    write_full_pae = False
    write_full_pde = False
    use_potentials = False

    def build_model() -> Boltz2ChunkInfer:
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
        steering_args.fk_steering = use_potentials
        steering_args.physical_guidance_update = use_potentials

        model = Boltz2TrunkInfer.load_from_checkpoint(
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

    def run_single_step(active_model: Boltz2TrunkInfer) -> None:
        with torch.inference_mode():
            out = active_model(batch, recycling_steps=args.recycling_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return out

    for mode in compile_modes:
        model = build_model()
        if should_print_model:
            print(model)
            should_print_model = False

        normalized_mode = mode.lower()
        model_to_run = model
        compile_elapsed = 0.0

        if normalized_mode != "eager":
            compile_start = time.time()
            model_to_run = torch.compile(
                model,
                mode=normalized_mode,
                dynamic=False,
            )
            compile_elapsed = time.time() - compile_start
            print(f"[compile:{mode}] torch.compile finished in {compile_elapsed:.2f}s")
        else:
            print(f"[compile:{mode}] running without torch.compile")

        warmup_elapsed = 0.0
        model_to_run.eval()
        if args.warmup_iters > 0:
            warmup_start = time.time()
            for _ in range(args.warmup_iters):
                _ = run_single_step(model_to_run)
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
                out = run_single_step(model_to_run)
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

        trace_label = normalized_mode
        trace_path = Path(f"trace-json-{trace_label}.json")
        prof.export_chrome_trace(trace_path.as_posix())
        print(f"[profile:{mode}] Chrome trace exported to {trace_path.resolve()}")

        del model_to_run
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(out["pdistogram"][0,0,0,0,0])
        print(out["pdistogram"].shape)